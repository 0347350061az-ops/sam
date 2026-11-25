"""Fast recovery boxes; category detection never confirms target identity."""
from pathlib import Path
import cv2
import torch
from recovery.proposals import overlap


class ObjectDiscovery:
    def __init__(self):
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
        path=Path(__file__).resolve().parents[1]/'data/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
        if not path.is_file():
            raise FileNotFoundError(f'Object discovery weights missing: {path}')
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=ssdlite320_mobilenet_v3_large(weights=None,weights_backbone=None).eval().to(self.device)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        with torch.inference_mode():
            self.model([torch.zeros((3,320,320),device=self.device)])
        # The camera worker subsequently uses its own CUDA stream.
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def detect(self,frame):
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        tensor=torch.from_numpy(rgb).to(self.device).permute(2,0,1).float()/255.
        result=self.model([tensor])[0]
        keep=result['scores']>=.25
        boxes=result['boxes'][keep].cpu().tolist()
        scores=result['scores'][keep].cpu().tolist()
        labels=result['labels'][keep].cpu().tolist()
        return [dict(box=b,score=s,label=int(label),detector=True,rotation=0)
                for b,s,label in zip(boxes,scores,labels)]

    def selected_class(self,frame,box):
        ranked=sorted(self.detect(frame),key=lambda c:overlap(c['box'],box),reverse=True)
        if ranked and overlap(ranked[0]['box'],box)>=.4:
            return ranked[0]['label']
        return None

    def proposals(self,frame,label):
        return [c for c in self.detect(frame) if c['label']==label]

    @torch.inference_mode()
    def embeddings(self,frame,boxes):
        """Class-agnostic crop descriptors from the already loaded backbone."""
        import torch.nn.functional as F
        tensors=[]
        h,w=frame.shape[:2]
        for box in boxes:
            x,y,r,b=map(int,box)
            x,y=max(0,min(w-1,x)),max(0,min(h-1,y))
            r,b=max(x+1,min(w,r)),max(y+1,min(h,b))
            rgb=cv2.cvtColor(frame[y:b,x:r],cv2.COLOR_BGR2RGB)
            tensors.append(torch.from_numpy(rgb).to(self.device).permute(2,0,1).float()/255.)
        if not tensors:return []
        batch,_=self.model.transform(tensors,None)
        features=self.model.backbone(batch.tensors)
        pooled=[F.normalize(F.adaptive_avg_pool2d(v,1).flatten(1),dim=1)
                for v in list(features.values())[:2]]
        return F.normalize(torch.cat(pooled,dim=1),dim=1).cpu().numpy()
