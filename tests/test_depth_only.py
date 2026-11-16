import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from corruption.mixture import floodDataset, singleDEMFloodDataset, yilanDataset, hsinchuDataset
from guided_diffusion.script_util import create_model
from i2sb.network import Image256Net
from i2sb.runner import Runner
from i2sb.util import custom_collate_fn
from sample import compute_batch, depth_to_image, save_depth_image


class Log:
    def info(self, *args): pass
    def warning(self, *args): pass


class DepthOnlyTests(unittest.TestCase):
    def test_single_dem_without_velocity_or_guidance_files(self):
        reads = []

        def read_image(path, flags):
            reads.append(str(path))
            self.assertNotIn('\\Vx', str(path))
            self.assertNotIn('\\Vy', str(path))
            if 'tainan_dem' in str(path):
                return np.full((256, 256, 3), 128, dtype=np.uint8)
            return np.full((256, 256), 250, dtype=np.uint8)

        def exists(path):
            return 'depth' in str(path)

        opt = SimpleNamespace(spm=False, ca4d=False)
        with patch('corruption.mixture.pd.read_csv', return_value=pd.DataFrame({'rf_1': [10] * 25})), \
             patch('corruption.mixture.os.path.exists', side_effect=exists), \
             patch('corruption.mixture.cv2.imread', side_effect=read_image), \
             contextlib.redirect_stdout(io.StringIO()):
            dataset = singleDEMFloodDataset(opt)
            self.assertEqual(len(dataset), 25)
            item = dataset[0]
        self.assertEqual(tuple(item[0].shape), (1, 256, 256))
        self.assertIsNone(item[1])
        self.assertIsNone(item[2])
        batch = custom_collate_fn([item, item])
        self.assertIsNone(batch[1])
        self.assertIsNone(batch[2])
        options = SimpleNamespace(device='cpu', add_x1_noise=False, cond_x1=True, spm=False, ca4d=False)
        training = Runner.__new__(Runner).sample_batch(options, iter([batch]), None)
        sampling = compute_batch(options, 'mixture', None, batch)
        self.assertEqual(training[0].shape, (2, 1, 256, 256))
        torch.testing.assert_close(training[0], sampling[0])
        torch.testing.assert_close(training[1], sampling[1])

    def test_other_datasets_do_not_read_velocity_targets(self):
        for cls in (floodDataset, yilanDataset, hsinchuDataset):
            with self.subTest(dataset=cls.__name__):
                ds = cls.__new__(cls)
                ds.opt = SimpleNamespace()
                ds.data_root = Path('unused')
                ds.test = False
                ds.use_spm = False
                ds.use_ca4d = True
                ds.transform = transforms.ToTensor()
                ds.spm_values = [10]
                ds.terrain_depths = {1: 4.0, 'yilan01': 4.0, 'hsinchu01': 4.0}
                ds.flood_path = ds.vx_path = ds.vy_path = 'unused'
                ds.dem_folder = ds.tuflow_d_folder = ds.tuflow_vx_folder = ds.tuflow_vy_folder = 'unused'
                ds.spm_folder = ds.ca4d_folder = 'unused'
                ds.dem_stat = pd.DataFrame({'Filename': [1], 'Min Elevation': [0], 'Max Elevation': [10]})
                ds.dem_info = {'yilan01': {'min': 0, 'max': 10}, 'hsinchu01': {'min': 0, 'max': 10}}
                if cls is floodDataset:
                    ds.cell_positions = [(1, 1, 24)]
                    ds.rainfall = [[10] * 24]
                else:
                    ds.cell_positions = [('yilan01' if cls is yilanDataset else 'hsinchu01', 1, 24)]
                    ds.rainfall_values = [np.array([10] * 24)]

                def read_image(path, flags):
                    self.assertNotIn('_vx_', str(path))
                    self.assertNotIn('_vy_', str(path))
                    return np.full((256, 256), 250, dtype=np.uint8)

                with patch('corruption.mixture.cv2.imread', side_effect=read_image), patch('corruption.mixture.os.path.exists', return_value=True), contextlib.redirect_stdout(io.StringIO()):
                    item = ds[0]
                self.assertEqual(item[0].shape, (1, 256, 256))
                self.assertIsNone(item[1])
                self.assertIsNone(item[2])
                self.assertEqual(item[12].shape, (1, 256, 256))

    def test_network_depth_loss_and_guidance_gradients(self):
        # Exercise real UNet/attention code at a smaller width for a fast regression test.
        def small_model(**kwargs):
            kwargs.update(image_size=64, num_channels=32, num_res_blocks=1, channel_mult='1,1')
            return create_model(**kwargs)

        torch.set_num_threads(2)
        for guidance in ('none', 'spm', 'ca4d'):
            with self.subTest(guidance=guidance), patch('i2sb.network.create_model', side_effect=small_model):
                net = Image256Net(Log(), torch.arange(4), spm=guidance == 'spm', ca4d=guidance == 'ca4d')
                x = torch.randn(1, 1, 16, 16)
                inputs = {guidance: torch.randn(1, 1, 16, 16)} if guidance != 'none' else {}
                pred = net(x, torch.tensor([1]), torch.randn(1, 24, 256), **inputs)
                self.assertEqual(pred.shape, x.shape)
                loss = (pred - torch.randn_like(pred)).square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                if guidance != 'none':
                    grad = getattr(net, guidance + '_adapter').weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)

    def test_depth_denormalization_matches_active_datasets(self):
        for path, single, mean, std in (
            ('RF01_d_001.png', True, .987, .0343),
            ('1_RF01_d_001.png', False, .986, .0405),
            ('yilan01_rf01_d_001.png', False, .9880, .0279),
            ('hsinchu01_rf01_d_001.png', False, .9751028, .0256081),
        ):
            raw = torch.tensor([[[.9]]])
            torch.testing.assert_close(depth_to_image((raw - mean) / std, path, single), raw)

    def test_training_checkpoint_sampling_and_grayscale_export(self):
        def small_model(**kwargs):
            kwargs.update(image_size=64, num_channels=32, num_res_blocks=1, channel_mult='1,1')
            return create_model(**kwargs)

        depth = torch.randn(1, 16, 16)
        item = (depth, None, None, torch.randn_like(depth), None, None, None,
                torch.ones(24, dtype=torch.long), 'RF01_d_001_00.png', None, None,
                None, torch.randn_like(depth), None, 4., 1)
        batch = custom_collate_fn([item])
        with tempfile.TemporaryDirectory() as folder, \
             patch('i2sb.network.create_model', side_effect=small_model), \
             patch('i2sb.util.build_log_writer', return_value=Mock()), \
             patch('i2sb.util.setup_loader', side_effect=lambda *args: iter([batch])):
            opt = SimpleNamespace(
                device='cpu', latent_space=False, normalize_latent=False, use_physics=False,
                load=None, ckpt_path=Path(folder), interval=4, beta_max=.1, t0=.0001, T=1.,
                use_fp16=False, cond_x1=True, ca4d=True, spm=False, ema=.99,
                lr=.0001, l2_norm=0., lr_gamma=1., lr_step=1, microbatch=1,
                batch_size=1, global_size=1, global_rank=0, num_itr=1,
                timestep_importance='continuous', ot_ode=False, add_x1_noise=False,
                distributed=False,
            )
            runner = Runner(opt, Log())
            before = runner.net.ca4d_adapter.weight.detach().clone()
            runner.train(opt, [item], [item], None)
            self.assertFalse(torch.equal(before, runner.net.ca4d_adapter.weight))
            self.assertTrue((Path(folder) / 'options.pkl').exists())
            opt.load = Path(folder) / 'latest.pt'
            restored = Runner(opt, Log(), save_opt=False)
            xs, _ = restored.ddpm_sampling(opt, batch[3], batch[7], nfe=2, verbose=False,
                                           ode_method='euler-maruyama', ca4d=batch[12])
            self.assertEqual(xs.shape[0], 1)
            self.assertEqual(tuple(xs.shape[2:]), (1, 16, 16))
            self.assertTrue(torch.isfinite(xs).all())
            save_depth_image(xs[0, 0], item[8], folder, single_dem=True)
            from PIL import Image
            images = list(Path(folder).glob('*.png'))
            self.assertEqual([p.name for p in images], ['recon_RF01_d_001_00.png'])
            with Image.open(images[0]) as image:
                self.assertEqual(image.mode, 'L')

    def test_unsupported_modes_and_old_checkpoint_fail_early(self):
        for mode in ('use_physics', 'latent_space'):
            options = SimpleNamespace(use_physics=False, latent_space=False)
            setattr(options, mode, True)
            with self.assertRaisesRegex(ValueError, 'Depth-only'):
                Runner(options, Log(), save_opt=False)

    def test_dem_missing_reports_path_and_grayscale_dem_loads(self):
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            options = SimpleNamespace(dataset_dir=root, spm=False, ca4d=False)
            with self.assertRaisesRegex(FileNotFoundError, 'tainan_dem.png.*--dataset-dir'):
                singleDEMFloodDataset(options)
            pd.DataFrame({'rf_1': [10] * 25}).to_csv(root / 'train_rf.csv', index=False)
            depth_folder = root / 'train' / 'depth' / 'RF01'
            depth_folder.mkdir(parents=True)
            cv2.imwrite(str(depth_folder / 'RF01_d_000_00.png'), np.full((16, 16), 250, dtype=np.uint8))
            for shape in ((16, 16), (16, 16, 3)):
                cv2.imwrite(str(root / 'tainan_dem.png'), np.full(shape, 128, dtype=np.uint8))
                with contextlib.redirect_stdout(io.StringIO()):
                    dataset = singleDEMFloodDataset(options)
                self.assertEqual(dataset.dem.shape, (16, 16))
                self.assertEqual(dataset.data_root, root.resolve())
                self.assertEqual(len(dataset), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'old.pt'
            torch.save({'net': {'diffusion_model.input_blocks.0.0.weight': torch.zeros(2, 3, 3, 3)}}, path)
            options = SimpleNamespace(use_physics=False, latent_space=False, load=path)
            with self.assertRaisesRegex(ValueError, 'Three-channel checkpoint'):
                Runner(options, Log(), save_opt=False)


if __name__ == '__main__':
    unittest.main()
