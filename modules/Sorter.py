import pagexml
import torch.nn as nn

class RoISorter(nn.Module):
    # Module to sort detected Regions of Interest by reading order (left-right,top-bottom)
    def __init__(self):
        super(SortRois,self).__init__()
    
    def forward(self,rois):
        pxml = pagexml.PageXML()

        pxml.newXml('retinanet_dets','image',1200,800)
        page = pxml.selectNth("//_:Page",0)
        reg = pxml.addTextRegion(page)
        pxml.setCoordsBBox(reg,0, 0, 1200,800 )
        line = pxml.addTextLine(reg)
        pxml.setCoordsBBox(line,0, 0, 1200,800 )
        for roi in range(rois.shape[0]):
            
            word = pxml.addWord(line)
            x1=int(rois[roi,0].cpu().detach())
            y1=int(rois[roi,1].cpu().detach())

            x2=int(rois[roi,2].cpu().detach().int())

            y2=int(rois[roi,3].cpu().detach().int())

            # Set text region bounding box with a confidence
            pxml.setCoordsBBox(word,x1, y1, x2-x1, y2-y1)
        words=pxml.select('//_:Word')
        order, groups = pxml.getLeftRightTopBottomReadingOrder(words, fake_baseline=True, max_horiz_iou=1, prolong_alpha=0.0)
        line = pxml.selectNth('//_:TextLine')
        group_idx = 0
        idx_in_group=0
        for n in order:
            word_idx = order.index(n)
            if idx_in_group>=groups[group_idx]:
                group_idx+=1
                idx_in_group=0

            pxml.setProperty(words[n],'word_idx',str(word_idx))
            pxml.setProperty(words[n],"line",str(group_idx))
            pxml.moveElem(words[n],line)
            idx_in_group+=1

        return order    
