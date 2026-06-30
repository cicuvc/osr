import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _discover_sarlo_pairs(data_root: str) -> list[tuple[str, str]]:
    pairs = []
    for fname in sorted(os.listdir(data_root)):
        if fname.endswith(".optic.png"):
            prefix = fname[: -len(".optic.png")]
            sar_name = f"{prefix}.sar.png"
            if os.path.exists(os.path.join(data_root, sar_name)):
                pairs.append(("", prefix))
    return pairs


def _discover_ubc_pairs(data_root: str) -> list[tuple[str, str]]:
    pairs = []
    rgb_dir = os.path.join(data_root, "rgb")
    sar_dir = os.path.join(data_root, "sar")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(sar_dir):
        return pairs
    for fname in sorted(os.listdir(rgb_dir)):
        if fname.endswith((".tif", ".tiff", ".png", ".jpg")):
            if os.path.exists(os.path.join(sar_dir, fname)):
                pairs.append((fname,))
    return pairs


def _discover_pairs(data_root: str) -> list[tuple[str, str]]:
    pairs = []
    for subdir in sorted(os.listdir(data_root)):
        subpath = os.path.join(data_root, subdir)
        if not os.path.isdir(subpath):
            continue
        for fname in os.listdir(subpath):
            if fname.startswith("opt") and fname.endswith(".png"):
                num_str = fname[3:-4]
                sar_name = f"sar{num_str}.png"
                if os.path.exists(os.path.join(subpath, sar_name)):
                    pairs.append((subdir, num_str))
    return pairs


def _discover_soma_pairs(data_root: str) -> list[tuple[str, str]]:
    pairs = []
    for satellite in sorted(os.listdir(data_root)):
        sat_path = os.path.join(data_root, satellite)
        if not os.path.isdir(sat_path):
            continue
        sar_dir = os.path.join(sat_path, "L")
        opt_dir = os.path.join(sat_path, "R")
        if not os.path.isdir(sar_dir) or not os.path.isdir(opt_dir):
            continue
        for fname in os.listdir(sar_dir):
            if fname.endswith(".png"):
                if os.path.exists(os.path.join(opt_dir, fname)):
                    pairs.append((satellite, fname))
    return pairs


def _discover_3mos_pairs(data_root: str) -> list[tuple[str, str]]:
    pairs = []
    for satellite in sorted(os.listdir(data_root)):
        sat_path = os.path.join(data_root, satellite)
        if not os.path.isdir(sat_path):
            continue
        for root, dirs, files in os.walk(sat_path):
            if os.path.basename(root) == "sar":
                sar_dir = root
                parent_dir = os.path.dirname(root)
                opt_base = os.path.join(parent_dir, "opt")
                if not os.path.isdir(opt_base):
                    continue
                for fname in sorted(files):
                    if not fname.startswith("sar_"):
                        continue
                    for ext in (".jpg", ".png"):
                        if fname.endswith(ext):
                            num_id = fname[len("sar_") : -len(ext)]
                            opt_fname = f"opt_{num_id}{ext}"
                            for scene in os.listdir(opt_base):
                                scene_path = os.path.join(opt_base, scene)
                                if not os.path.isdir(scene_path):
                                    continue
                                if os.path.exists(os.path.join(scene_path, opt_fname)):
                                    sar_rel = os.path.relpath(
                                        os.path.join(sar_dir, fname), data_root
                                    )
                                    opt_rel = os.path.relpath(
                                        os.path.join(scene_path, opt_fname), data_root
                                    )
                                    pairs.append((sar_rel, opt_rel))
                                    break
                    else:
                        continue
                    break
    return pairs


def _compute_iou(
    patch_size: int,
    theta_rad: float,
    dx: float,
    dy: float,
) -> float:
    L = patch_size
    xs = torch.arange(L, dtype=torch.float32)
    ys = torch.arange(L, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    center = (L - 1) / 2.0
    cos_th = math.cos(theta_rad)
    sin_th = math.sin(theta_rad)

    x_rot = cos_th * (xx - center) + sin_th * (yy - center) + center - (cos_th * dx + sin_th * dy)
    y_rot = -sin_th * (xx - center) + cos_th * (yy - center) + center - (-sin_th * dx + cos_th * dy)

    mask_a = (x_rot >= 0) & (x_rot < L) & (y_rot >= 0) & (y_rot < L)
    intersection = mask_a.sum().item()
    area = L * L
    union = 2 * area - intersection
    if union == 0:
        return 0.0
    return intersection / union


def _box_a_in_bounds(
    xb: int,
    yb: int,
    theta_rad: float,
    dx: float,
    dy: float,
    W: int,
    H: int,
    L: int,
) -> bool:
    center = (L - 1) / 2.0
    cA_x = xb + center + dx
    cA_y = yb + center + dy
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    signs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    for sx, sy in signs:
        cx = cA_x + center * (sx * cos_t - sy * sin_t)
        cy = cA_y + center * (sx * sin_t + sy * cos_t)
        if cx < 0 or cx > W - 1 or cy < 0 or cy > H - 1:
            return False
    return True


def make_dataset(
    data_roots: list[str],
    fmts: list[str] | None = None,
    patch_size: int = 384,
    max_angle_deg: float = 30.0,
    max_translation: float = 64.0,
    iou_thresh: float = 0.7,
    seed: int | None = None,
    max_pairs: int | None = None,
    subset: str | None = None,
) -> torch.utils.data.ConcatDataset:
    if fmts is None:
        fmts = ["osd"] * len(data_roots)
    datasets = []
    for root, fmt in zip(data_roots, fmts):
        ds = OpticalSARDataset(
            data_root=root,
            patch_size=patch_size,
            max_angle_deg=max_angle_deg,
            max_translation=max_translation,
            iou_thresh=iou_thresh,
            seed=seed,
            max_pairs=max_pairs,
            fmt=fmt,
            subset=subset,
        )
        datasets.append(ds)
    return torch.utils.data.ConcatDataset(datasets)


class OpticalSARDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        patch_size: int = 384,
        max_angle_deg: float = 30.0,
        max_translation: float = 64.0,
        iou_thresh: float = 0.7,
        seed: int | None = None,
        max_pairs: int | None = None,
        fmt: str = "osd",
        subset: str | None = None,
        img_resize: int | None = None,
        img_padding: bool = True,
        df: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.data_root = data_root
        self.patch_size = patch_size
        self.max_angle_deg = max_angle_deg
        self.max_translation = max_translation
        self.iou_thresh = iou_thresh
        self.rng = np.random.default_rng(seed)
        self.fmt = fmt
        self.img_resize = img_resize
        self.img_padding = img_padding
        self.df = df
        self.coarse_scale = kwargs.get('coarse_scale', None)

        if fmt == "soma":
            all_pairs = _discover_soma_pairs(data_root)
        elif fmt == "3mos":
            all_pairs = _discover_3mos_pairs(data_root)
        elif fmt == "sarlo":
            all_pairs = _discover_sarlo_pairs(data_root)
        elif fmt == "ubc":
            all_pairs = _discover_ubc_pairs(data_root)
        else:
            all_pairs = _discover_pairs(data_root)

        if max_pairs is not None:
            all_pairs = all_pairs[:max_pairs]

        self.pairs = self._apply_partition(all_pairs, data_root, subset)
        self.subset = subset

    def _apply_partition(self, all_pairs, data_root, subset):
        partition_path = os.path.join(data_root, "partition.json")

        if os.path.exists(partition_path):
            with open(partition_path) as f:
                partition = json.load(f)
        else:
            rng = np.random.default_rng(42)
            indices = list(range(len(all_pairs)))
            rng.shuffle(indices)
            n = len(indices)
            n_train = int(n * 0.7)
            n_val = int(n * 0.1)
            partition = {
                "train": indices[:n_train],
                "val": indices[n_train : n_train + n_val],
                "test": indices[n_train + n_val :],
            }
            os.makedirs(data_root, exist_ok=True)
            with open(partition_path, "w") as f:
                json.dump(partition, f)

        if subset is not None:
            idxs = partition.get(subset, [])
            return [all_pairs[i] for i in idxs]
        return all_pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, *args) -> tuple:
        if self.fmt == "soma":
            satellite, fname = args
            sar_path = os.path.join(self.data_root, satellite, "L", fname)
            opt_path = os.path.join(self.data_root, satellite, "R", fname)
        elif self.fmt == "3mos":
            sar_rel, opt_rel = args
            sar_path = os.path.join(self.data_root, sar_rel)
            opt_path = os.path.join(self.data_root, opt_rel)
        elif self.fmt == "sarlo":
            _, prefix = args
            sar_path = os.path.join(self.data_root, f"{prefix}.sar.png")
            opt_path = os.path.join(self.data_root, f"{prefix}.optic.png")
        elif self.fmt == "ubc":
            (fname,) = args
            sar_path = os.path.join(self.data_root, "sar", fname)
            opt_path = os.path.join(self.data_root, "rgb", fname)
        else:
            subdir, num_str = args
            sar_path = os.path.join(self.data_root, subdir, f"sar{num_str}.png")
            opt_path = os.path.join(self.data_root, subdir, f"opt{num_str}.png")

        sar_img = Image.open(sar_path)
        opt_img = Image.open(opt_path)
        if self.fmt == "ubc" and sar_img.mode == "F":
            sar_arr = np.array(sar_img, dtype=np.float32)
            sar_arr = np.clip(sar_arr / (sar_arr.max() + 1e-8), 0.0, 1.0)
        else:
            sar_arr = np.array(sar_img.convert("L"), dtype=np.float32) / 255.0
        opt_arr = np.array(opt_img.convert("L"), dtype=np.float32) / 255.0
        return (
            torch.from_numpy(sar_arr).unsqueeze(0),
            torch.from_numpy(opt_arr).unsqueeze(0),
            sar_path,
            opt_path,
        )

    def _augment_optic(self, img: torch.Tensor) -> torch.Tensor:
        if self.subset != 'train': # disable augment for eval/test
            return img

        p = self.rng.random()
        if p < 0.4:
            img = img * float(self.rng.uniform(0.85, 1.15))
        p = self.rng.random()
        if p < 0.4:
            contrast = float(self.rng.uniform(0.8, 1.2))
            img = (img - img.mean()) * contrast + img.mean()
        p = self.rng.random()
        if p < 0.3:
            img = img + torch.randn_like(img) * float(self.rng.uniform(0.0, 0.03))
        p = self.rng.random()
        if p < 0.2:
            gamma = float(self.rng.uniform(0.7, 1.3))
            img = img.clamp(min=1e-6) ** gamma
        return img.clamp(0.0, 1.0)

    def _compute_flow_and_latent(self, theta_rad: float, dx: float, dy: float) -> tuple:
        L = self.patch_size
        center = (L - 1) / 2.0
        cos_t = math.cos(-theta_rad)
        sin_t = math.sin(-theta_rad)
        A = complex(cos_t, sin_t)
        T_cmplx = complex(dx, dy) + complex(center, center)
        T_flow = complex(center, center) - A * T_cmplx
        xs = torch.arange(L, dtype=torch.float32)
        ys = torch.arange(L, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        z = xx + 1j * yy
        f_z = A * z + T_flow
        flow = torch.stack([f_z.real, f_z.imag], dim=-1)
        flow = flow.permute(2, 0, 1)
        r_re = torch.full((1, L, L), 0.0, dtype=torch.float32)
        r_im = torch.full((1, L, L), -theta_rad, dtype=torch.float32)
        r_gt = torch.cat([r_re, r_im], dim=0)
        s_re = torch.full((1, L, L), T_flow.real, dtype=torch.float32)
        s_im = torch.full((1, L, L), T_flow.imag, dtype=torch.float32)
        s_gt = torch.cat([s_re, s_im], dim=0)
        nu_gt = torch.zeros(2, L, L, dtype=torch.float32)
        x0 = torch.cat([nu_gt, r_gt, s_gt], dim=0)
        return flow, x0

    def _sample_crop_and_transform(self, full_h: int, full_w: int) -> tuple:
        L = self.patch_size
        max_attempts = 500
        for _ in range(max_attempts):
            xb = int(self.rng.integers(0, full_w - L + 1))
            yb = int(self.rng.integers(0, full_h - L + 1))
            theta_deg = self.rng.uniform(-self.max_angle_deg, self.max_angle_deg)
            theta_rad = theta_deg * math.pi / 180.0
            dx = self.rng.uniform(-self.max_translation, self.max_translation)
            dy = self.rng.uniform(-self.max_translation, self.max_translation)
            if _compute_iou(L, theta_rad, dx, dy) < self.iou_thresh:
                continue
            if not _box_a_in_bounds(xb, yb, theta_rad, dx, dy, full_w, full_h, L):
                continue
            return xb, yb, theta_rad, dx, dy
        xb = int(self.rng.integers(0, full_w - L + 1))
        yb = int(self.rng.integers(0, full_h - L + 1))
        return xb, yb, 0.0, 0.0, 0.0

    def _augment_geometry(
        self, sar: torch.Tensor, opt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.subset != 'train': # disable augment for eval/test
            return sar, opt
        p = self.rng.random()
        if p < 0.3:
            phi = float(self.rng.uniform(0.0, 360.0))
            phi_rad = phi * math.pi / 180.0
            cos_p = math.cos(phi_rad)
            sin_p = math.sin(phi_rad)
            theta = torch.tensor(
                [[cos_p, -sin_p, 0.0], [sin_p, cos_p, 0.0]],
                dtype=torch.float32,
            )
            H, W = sar.shape[-2], sar.shape[-1]
            grid = torch.nn.functional.affine_grid(
                theta.unsqueeze(0),
                (1, 1, H, W),
                align_corners=True,
            )
            sar = torch.nn.functional.grid_sample(
                sar.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)
            opt = torch.nn.functional.grid_sample(
                opt.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)

            abs_cos = abs(cos_p)
            abs_sin = abs(sin_p)
            crop_scale = 1.0 / (abs_cos + abs_sin)
            crop_h = int(H * crop_scale) & ~1
            crop_w = int(W * crop_scale) & ~1
            y0 = (H - crop_h) // 2
            x0 = (W - crop_w) // 2
            sar = sar[:, y0 : y0 + crop_h, x0 : x0 + crop_w]
            opt = opt[:, y0 : y0 + crop_h, x0 : x0 + crop_w]
            sar = F.interpolate(
                sar.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            opt = F.interpolate(
                opt.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        p = self.rng.random()
        if p < 0.3:
            sar = torch.flip(sar, dims=(-1,))
            opt = torch.flip(opt, dims=(-1,))
        p = self.rng.random()
        if p < 0.3:
            sar = torch.flip(sar, dims=(-2,))
            opt = torch.flip(opt, dims=(-2,))
        return sar, opt

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        sar_full, opt_full, sar_path, opt_path = self._load_image(*pair)

        sar_full, opt_full = self._augment_geometry(sar_full, opt_full)

        full_h, full_w = sar_full.shape[-2], sar_full.shape[-1]
        L = self.patch_size

        xb, yb, theta_rad, dx, dy = self._sample_crop_and_transform(full_h, full_w)

        sar_patch = sar_full[:, yb : yb + L, xb : xb + L]

        center_a_x = xb + (L - 1) / 2.0 + dx
        center_a_y = yb + (L - 1) / 2.0 + dy
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)

        xs = torch.arange(L, dtype=torch.float32)
        ys = torch.arange(L, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        abs_x = center_a_x + cos_t * (xx - (L - 1) / 2.0) - sin_t * (yy - (L - 1) / 2.0)
        abs_y = center_a_y + sin_t * (xx - (L - 1) / 2.0) + cos_t * (yy - (L - 1) / 2.0)

        opt_crop = opt_full
        grid_x = 2.0 * abs_x / (full_w - 1) - 1.0
        grid_y = 2.0 * abs_y / (full_h - 1) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        opt_patch = torch.nn.functional.grid_sample(
            opt_crop.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)

        opt_patch = self._augment_optic(opt_patch)

        flow, x0 = self._compute_flow_and_latent(theta_rad, dx + 6, dy)

        if self.img_resize is not None:
            H0, W0 = sar_patch.shape[-2], sar_patch.shape[-1]
            scale_factor = self.img_resize / max(H0, W0)
            H_new = int(round(H0 * scale_factor))
            W_new = int(round(W0 * scale_factor))
            if self.img_padding:
                h_pad = self.df - H_new % self.df if H_new % self.df != 0 else 0
                w_pad = self.df - W_new % self.df if W_new % self.df != 0 else 0
                pad_h = max(H_new + h_pad, W_new + w_pad)
                pad_w = pad_h
            else:
                pad_h, pad_w = H_new, W_new

            sar_patch = F.interpolate(sar_patch[None], size=(H_new, W_new), mode='bilinear', align_corners=False)[0]
            opt_patch = F.interpolate(opt_patch[None], size=(H_new, W_new), mode='bilinear', align_corners=False)[0]
            flow = F.interpolate(flow[None], size=(H_new, W_new), mode='bilinear', align_corners=True)[0]
            flow[0] = flow[0] * (W_new / W0)
            flow[1] = flow[1] * (H_new / H0)

            if self.img_padding:
                sar_patch = F.pad(sar_patch, (0, w_pad, 0, h_pad))
                opt_patch = F.pad(opt_patch, (0, w_pad, 0, h_pad))
                flow = F.pad(flow, (0, w_pad, 0, h_pad))
                if pad_w > W_new + w_pad:
                    sar_patch = F.pad(sar_patch, (0, pad_w - (W_new + w_pad), 0, 0))
                    opt_patch = F.pad(opt_patch, (0, pad_w - (W_new + w_pad), 0, 0))
                    flow = F.pad(flow, (0, pad_w - (W_new + w_pad), 0, 0))
                if pad_h > H_new + h_pad:
                    sar_patch = F.pad(sar_patch, (0, 0, 0, pad_h - (H_new + h_pad)))
                    opt_patch = F.pad(opt_patch, (0, 0, 0, pad_h - (H_new + h_pad)))
                    flow = F.pad(flow, (0, 0, 0, pad_h - (H_new + h_pad)))
                mask = torch.zeros(pad_h, pad_w, dtype=torch.bool)
                mask[:H_new, :W_new] = True
            else:
                mask = torch.ones(H_new, W_new, dtype=torch.bool)
            L = pad_h
        else:
            mask = torch.ones(L, L, dtype=torch.bool)

        scale = torch.tensor([1.0, 1.0], dtype=torch.float32)

        pair_names = (os.path.basename(opt_path), os.path.basename(sar_path))

        return_dict = {
            "image0": opt_patch,
            "image1": sar_patch,
            "image0_path": opt_path,
            "image1_path": sar_path,
            "flow": flow,
            "scale0": scale,
            "scale1": scale,
            "dataset_name": "OpticalSARDataset",
            "pair_id": idx,
            "pair_names": pair_names,
            "x0": x0,
            "theta_rad": theta_rad,
            "dx": dx,
            "dy": dy,
            "xb": xb,
            "yb": yb,
            "sample_id": "/".join(pair),
        }

        if self.coarse_scale:
            H, W = mask.shape
            coarse_h, coarse_w = int(H * self.coarse_scale), int(W * self.coarse_scale)
            mask_coarse = F.interpolate(
                mask[None, None].float(),
                size=(coarse_h, coarse_w),
                mode="nearest",
            )[0, 0].bool()
            return_dict["mask0"] = mask_coarse
            return_dict["mask1"] = mask_coarse

        return return_dict
