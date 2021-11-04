from validiation import Validation
class Fusion(object):
    def __init__(self,root,ckpt):
        self.val=Validation(root,ckpt)
    
    def calbration(self,):
        self.val.val()
        return NotImplementedError

if __name__ == "__main__":
    root="./data/waymo/waymo_processed_data/"
    sequece='segment-1024360143612057520_3580_000_3600_000_with_camera_labels'
    ckpt="./checkpoints/checkpoint_epoch_30.pth"
    fuse=Fusion(root,ckpt)