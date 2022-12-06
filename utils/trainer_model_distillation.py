import numpy as np
import torch
import os

import sys
sys.path.append("../")

from utils.evaluator import Evaluator


class TrainerDistillation:
    def __init__(self, model, 
                       teacher_model,
                       distill_criterion, 
                       student_criterion,
                       alpha_loss,
                       optimizer, 
                       config, 
                       ckpt_save_path=None, 
                       scheduler=None):
        self.model = model
        self.distill_criterion = distill_criterion
        self.student_criterion = student_criterion
        self.alpha_loss = alpha_loss

        self.optimizer = optimizer

        self.distill_loss = {"train": [], "val": []}
        self.student_loss = {"train": [], "val": []}
        self.total_loss   = {"train": [], "val": []}
        self.test_rmse = []

        self.epochs = config["epochs"]
        self.batches_per_epoch = config["batches_per_epoch"]
        self.batches_per_epoch_val = config["batches_per_epoch_val"]
        self.device = config["device"]
        self.scheduler = scheduler
        self.checkpoint_frequency = 1
        self.early_stopping_epochs = 10
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5
        self.ckpt_save_path = ckpt_save_path

        if self.ckpt_save_path:
            os.makedirs(self.ckpt_save_path, exist_ok=True)

        self.teacher_model = teacher_model
        self.teacher_model = self._freeze_model_layers(self.teacher_model)

            



    
    def _freeze_model_layers(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def train(self, train_dataloader, val_dataloader, test_dataloader, load_chkpt=None):
        
        start_epoch = 0

        rmse_teacher, exec_time_avg = Evaluator.inference_fwd_baseline(self.teacher_model, test_dataloader)
        print("RMSE TEACHER BEFORE TRAINING", rmse_teacher)

        if load_chkpt:
            checkpoint = torch.load(load_chkpt)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1


        for epoch in range(start_epoch, self.epochs):
            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)

            rmse, exec_time_avg = Evaluator.inference_fwd_baseline(self.model, test_dataloader)
            # rmse_teacher, exec_time_avg = Evaluator.inference_fwd_baseline(self.teacher_model, test_dataloader)

            self.test_rmse.append(rmse)

            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}, Train Distill Loss={},  Val Distill Loss={}, Train Student Loss={}, Val Student Loss={}, Test RMSE={}, Test RMSE Teacher={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.total_loss["train"][-1], 10),
                    np.round(self.total_loss["val"][-1], 10),
                    np.round(self.distill_loss["train"][-1], 10),
                    np.round(self.distill_loss["val"][-1], 10),
                    np.round(self.student_loss["train"][-1], 10),
                    np.round(self.student_loss["val"][-1], 10),
                    self.test_rmse[-1],
                    rmse_teacher
                )
            )

            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(self.total_loss["train"][-1])


            if self.ckpt_save_path:
                save_path = f"{self.ckpt_save_path}/epoch_{epoch}"

                scheduler_state_dict = None
                
                if self.scheduler:
                    scheduler_state_dict = self.scheduler.state_dict()
                
                torch.save({ 
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': scheduler_state_dict,
                    'train_loss_list': self.total_loss["train"],
                    'val_loss_list':self.total_loss["val"],
                    'train_distill_loss_list': self.distill_loss["train"],
                    'val_distill_loss_list': self.distill_loss["val"],
                    'train_student_loss_list': self.student_loss["train"],
                    'val_student_loss_list': self.student_loss["val"],
                    'val RMSE': self.test_rmse[-1]
                },  save_path)

            # # early stopping
            # if epoch < self.early_stopping_avg:
            #     min_val_loss = np.round(np.mean(self.total_loss["val"]), self.early_stopping_precision)
            #     no_decrease_epochs = 0

            # else:
            #     val_loss = np.round(
            #         np.mean(self.total_loss["val"][-self.early_stopping_avg:]), 
            #                         self.early_stopping_precision
            #     )
            #     if val_loss >= min_val_loss:
            #         no_decrease_epochs += 1
            #     else:
            #         min_val_loss = val_loss
            #         no_decrease_epochs = 0
            #         #print('New min: ', min_val_loss)

            # if no_decrease_epochs > self.early_stopping_epochs:
            #     print("Early Stopping")
            #     break

        torch.save(self.model.state_dict(), "model_final")
        return self.model

    def _epoch_train(self, dataloader):

        self.model = self.model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)

        self.model.train()
        running_loss = []
        running_distill_loss = []
        running_student_loss = []

        self.teacher_model.eval() # Weights are frozen anyways

        for i, data in enumerate(dataloader, 0):
            inputs = data["image"].to(self.device)
            labels = data["heatmaps"].to(self.device)

            y_teacher = self.teacher_model(inputs)
            y_student = self.model(inputs)

            loss_distill = self.distill_criterion(y_teacher, y_student)
            loss_student = self.student_criterion(y_student, labels)

            total_loss = (1 - self.alpha_loss)*loss_distill + self.alpha_loss * loss_student
            # total_loss = loss_student

            total_loss.backward()
            self.optimizer.step()

            running_loss.append(total_loss.item())
            running_distill_loss.append(loss_distill.item())
            running_student_loss.append(loss_student.item())

            if i == self.batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                epoch_distill_loss = np.mean(running_distill_loss)
                epoch_student_loss = np.mean(running_student_loss)

                self.total_loss["train"].append(epoch_loss)
                self.distill_loss["train"].append(epoch_distill_loss)
                self.student_loss["train"].append(epoch_student_loss)
                break

    def _epoch_eval(self, dataloader):

        self.model = self.model.to(self.device)
        self.teacher_model = self.teacher_model.to(self.device)

        self.model.eval()
        running_loss = []
        running_distill_loss = []
        running_student_loss = []
        
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs = data["image"].to(self.device)
                labels = data["heatmaps"].to(self.device)

                # print("heatmaps", labels)

                y_teacher = self.teacher_model(inputs)
                y_student = self.model(inputs)

                loss_distill = self.distill_criterion(y_teacher, y_student)
                loss_student = self.student_criterion(y_student, labels)

                total_loss = loss_distill + self.alpha_loss * loss_student

                running_loss.append(total_loss.item())
                running_distill_loss.append(loss_distill.item())
                running_student_loss.append(loss_student.item())

                if i == self.batches_per_epoch:
                    epoch_loss = np.mean(running_loss)
                    epoch_distill_loss = np.mean(running_distill_loss)
                    epoch_student_loss = np.mean(running_student_loss)

                    self.total_loss["val"].append(epoch_loss)
                    self.distill_loss["val"].append(epoch_distill_loss)
                    self.student_loss["val"].append(epoch_student_loss)
                    break
