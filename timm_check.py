import timm
# print(timm.list_models())
model_tmp = timm.create_model(
    'swin_base_patch4_window7_224',
    pretrained=True,
    features_only=True,
    img_size=(768,1152)
)
for i, info in enumerate(model_tmp.feature_info):
    print(f"Stage {i}: {info}")

