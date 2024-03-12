from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

@BBOX_CODERS.register_module()
class AreaPseudoCoder(BaseBBoxCoder):
    def encode(self):
        pass

    def decode_single(self, cls_scores, area_preds):

        cls_scores = cls_scores.sigmoid()
        scores, labels = cls_scores.max(-1)

        predictions_dict = {
            'areas': area_preds.detach().cpu().numpy(),
            'scores': scores.detach().cpu().numpy(),
            'labels': labels.detach().cpu().numpy(),
        }

        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_pts_preds = preds_dicts['all_pts_preds'][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_pts_preds[i]))
        return predictions_list
