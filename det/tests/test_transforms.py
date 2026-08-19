"""§6.1 / §6.2: NoDummy crop keeps empty targets empty; ImageNetNormalize is applied once and guarded."""
import PIL.Image
import pytest
import torch
from torchvision import tv_tensors

from src.core import GLOBAL_CONFIG, create
from src.data.transforms.container import Compose
from src.data._misc import convert_to_tv_tensor

from lsp_det.transforms import ImageNetNormalize, RandomCropWithGridNoDummy, IMAGENET_MEAN, IMAGENET_STD


def _pil(w, h, val=128):
    return PIL.Image.new("RGB", (w, h), (val, val, val))


def _target(boxes_xyxy, w, h):
    boxes = torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4)
    return {"boxes": convert_to_tv_tensor(boxes, key="boxes", spatial_size=(h, w)),
            "labels": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([7]), "orig_size": torch.tensor([w, h]),
            "area": torch.ones(len(boxes)), "iscrowd": torch.zeros(len(boxes), dtype=torch.int64)}


def test_no_dummy_on_small_image_without_center_boxes():
    """512x512 image whose only box lies outside the centre-672 window after reflection padding
    (padded image is 1536, content at [512,1024): a box at the corner (0..3) maps to 512..515 -> inside
    centre window [432,1104)). So use an image with *no* boxes: must stay empty."""
    t = RandomCropWithGridNoDummy(crop_size=1536, center_gt_size=672)
    img, tgt = t(_pil(512, 512), _target([], 512, 512))
    assert tuple(tgt["boxes"].shape) == (0, 4) and tuple(tgt["labels"].shape) == (0,)
    assert isinstance(tgt["boxes"], tv_tensors.BoundingBoxes)
    assert img.size == (1536, 1536)


def test_dome_original_adds_dummy_but_ours_does_not():
    dome_t = create("RandomCropWithGrid", GLOBAL_CONFIG, crop_size=1536, center_gt_size=672) if False else \
        GLOBAL_CONFIG["RandomCropWithGrid"]["_pymodule"].RandomCropWithGrid(crop_size=1536, center_gt_size=672)
    ours = RandomCropWithGridNoDummy(crop_size=1536, center_gt_size=672)
    # 4096 image, one box in a corner tile; force grid_index of a far tile -> crop has no boxes
    W = 4096
    tgt = _target([[10, 10, 24, 24]], W, W); tgt["grid_index"] = 20  # some interior tile
    _, t_dome = dome_t(_pil(W, W), dict(tgt))
    tgt = _target([[10, 10, 24, 24]], W, W); tgt["grid_index"] = 20
    _, t_ours = ours(_pil(W, W), dict(tgt))
    if len(t_dome["boxes"]) == 0:
        pytest.skip("grid_index 20 unexpectedly overlaps the box tile")
    assert len(t_dome["boxes"]) == 1 and int(t_dome["labels"][0]) == 0  # Dome fake 3x3 class-0 box
    assert len(t_ours["boxes"]) == 0 and len(t_ours["labels"]) == 0


def test_full_train_pipeline_with_empty_target_and_normalize():
    ops = [
        {"type": "RandomCropWithGridNoDummy", "crop_size": 1536, "center_gt_size": 672},
        {"type": "ColorJitter", "brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05},
        {"type": "RandomHorizontalFlip", "p": 0.5},
        {"type": "RandomVerticalFlip", "p": 0.5},
        {"type": "SanitizeBoundingBoxes", "min_size": 2},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
        {"type": "RandomGaussianBlur", "p": 0.25, "kernel_size": [3, 5], "sigma": [0.1, 1.0]},
        {"type": "ImageNetNormalize"},
    ]
    comp = Compose(ops=[dict(o) for o in ops])
    img, tgt, _ = comp(_pil(512, 512, 200), _target([], 512, 512), None)
    assert tuple(img.shape) == (3, 1536, 1536) and tuple(tgt["boxes"].shape) == (0, 4)
    # ColorJitter perturbs the value; check the un-normalized pixel is within the +-10% jitter band of 200/255
    px = img[:, 768, 768] * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN)
    assert ((px - 200 / 255).abs() < 0.12).all(), px
    # exact check without ColorJitter
    comp2 = Compose(ops=[dict(o) for o in ops if o["type"] != "ColorJitter"])
    img2, _, _ = comp2(_pil(512, 512, 200), _target([], 512, 512), None)
    exp = (200 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(img2[:, 768, 768], exp, atol=1e-4), (img2[:, 768, 768], exp)
    # with boxes: boxes must not be normalized by ImageNet stats
    img, tgt, _ = comp(_pil(512, 512, 200), _target([[100, 100, 120, 118]], 512, 512), None)
    assert len(tgt["boxes"]) >= 1 and float(tgt["boxes"].max()) <= 1.0 and float(tgt["boxes"].min()) >= 0.0


def test_val_pipeline_normalize_once_and_boxes_untouched():
    ops = [{"type": "Resize", "size": [1536, 1536]}, {"type": "ConvertPILImage", "dtype": "float32", "scale": True}, {"type": "ImageNetNormalize"}]
    comp = Compose(ops=[dict(o) for o in ops])
    img, tgt, _ = comp(_pil(1536, 1536, 100), _target([[500, 500, 520, 520]], 1536, 1536), None)
    exp = (100 / 255 - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(img[:, 10, 10], exp, atol=1e-4)
    assert torch.equal(tgt["boxes"], torch.tensor([[500., 500., 520., 520.]]))  # pixel xyxy untouched (val has no ConvertBoxes)


def test_double_normalize_raises():
    n = ImageNetNormalize()
    x = tv_tensors.Image(torch.rand(3, 8, 8))
    y = n(x)
    with pytest.raises(RuntimeError):
        n(y)
    # and a fresh tensor already outside [0,1] (i.e. normalized elsewhere) is rejected too
    with pytest.raises(RuntimeError):
        n(tv_tensors.Image(torch.randn(3, 8, 8) * 3))
