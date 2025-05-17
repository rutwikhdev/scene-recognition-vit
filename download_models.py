import os
import gdown

models = {
    "vit_70": {
        "link": "https://drive.google.com/file/d/1nBl246KJDEWDpOTjnfvSm2c-zZ_YnB-d/view?usp=drive_link",
        "output": "vit.pth",
    },
    "swin_66": {
        "link": "https://drive.google.com/file/d/106UL379Gkv292QbZzT1sBZpFi1IeVwyy/view?usp=drive_link",
        "output": "swin.pth",
    },
    "dinov2_80": {
        "link": "https://drive.google.com/file/d/1RBS8Bq8V3fuY3lGMxdA6S0xP-7Lq0NcY/view?usp=drive_link",
        "output": "dino.pth",
    },
}

for model in models.keys():
    trained_model = models.get(model)
    if not os.path.exists(trained_model.get("output")):
        gdown.download(
            url=trained_model.get("link"),
            output=trained_model.get("output"),
            fuzzy=True,
        )
