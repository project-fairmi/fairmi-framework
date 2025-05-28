import timm

model = timm.create_model(
    'hiera_small_224.mae',
    pretrained=True,
    num_classes=0,
    cache_dir='/scratch/unifesp/fairmi/dilermando.queiroz/fairmi-framework/.cache'
)