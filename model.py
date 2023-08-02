
from controllers import download_checkpoint, download_checkpoint_from_google_drive
from track_anything import TrackingAnything


def get_model(args: dict): 
    # check and download checkpoints if needed
    SAM_checkpoint_dict = {
        'vit_h': "sam_vit_h_4b8939.pth",
        'vit_l': "sam_vit_l_0b3195.pth", 
        "vit_b": "sam_vit_b_01ec64.pth"
    }
    SAM_checkpoint_url_dict = {
        'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    }
    sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
    sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
    xmem_checkpoint = "XMem-s012.pth"
    xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
    e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
    e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


    folder ="./checkpoints"
    SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
    xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
    e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
    args.port = 12212
    args.device = "cpu"
    # args.mask_save = True

    # initialize sam, xmem, e2fgvi models
    model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)

    return model

