# trainers/svrg.py
from trainers.base_svrg import BaseSVRGTrainer

class SVRGTrainer(BaseSVRGTrainer):
    def _run_epoch(self, data_loader, epoch, is_train=True):
        if is_train and (epoch % self.update_freq_epochs == 0):
            self._update_ref_and_mu()
        return super()._run_epoch(data_loader, epoch, is_train=is_train)