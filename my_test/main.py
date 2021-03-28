import os
import logging
from argparse import ArgumentParser
from datetime import datetime
import pdb
import torch
from torch import nn
from torch import optim
from torch.nn import DataParallel
from transformers import get_cosine_schedule_with_warmup,BertTokenizer

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage

from my_test.extract import BertTool
from my_test.models import BertClassificationModel

logger = logging.getLogger()

current_time = datetime.now().strftime('%Y%m%d%H%M%S')
checkpoint_dir = os.path.join('./checkpoint', current_time)


def train():
    logger.info('*' * 64)
    logger.info('token:%s' % current_time)
    logger.info('*' * 64)

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./my_test/data/nlpcc2018+hsk/",
                        help="Path or url of the dataset. If empty download from S3.")

    parser.add_argument("--dataset_cache", type=str, default='./cache/', help="Path or url of the dataset cache")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-4, help="Learning rate")
    # parser.add_argument("--train_precent", type=float, default=0.7, help="Batch size for validation")
    parser.add_argument("--n_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--log_step", type=int, default=1, help="Multiple-choice loss coefficient")
    parser.add_argument("--base_model", type=str, default="./my_test/model_files/bert/" )
    args = parser.parse_args()
    logger.info(args)
    device = torch.device(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.base_model)
    train_data_loader, valid_data_loader = BertTool.get_loaders(args.dataset_path,tokenizer,args.batch_size)

    model = BertClassificationModel(cls=tokenizer.vocab_size,model_file=args.base_model)



    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    steps = len(train_data_loader.dataset) // train_data_loader.batch_size
    steps = steps if steps > 0 else 1
    logger.info('steps:%d' % steps)

    lr_warmup = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1500, num_training_steps=steps*args.n_epochs)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        gpu_num = torch.cuda.device_count()
        gpu_list = [int(i) for i in range(gpu_num)]
        model = DataParallel(model, device_ids=gpu_list)
        multi_gpu = True

    if torch.cuda.is_available():
        model.cuda()

    # model.to(device)
    # criterion.to(device)

    def update(engine, batch):
        model.train()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        predict = output.permute(1,2,0)
        trg = labels.permute(1,0)
        loss = criterion(predict.to(device), trg.to(device))

        loss.backward()
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        lr_warmup.step()

        if multi_gpu:
            return loss.cpu().mean().item()
        return loss.cpu().item()

        return loss.item()

    trainer = Engine(update)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)

            predict = output.permute(1, 2, 0)
            trg = labels.permute(1, 0)
            # loss = criterion(predict.to(device), trg.to(device))

            return predict, trg

    evaluator = Engine(inference)
    metrics = {"nll": Loss(criterion, output_transform=lambda x: (x[0], x[1])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0], x[1]))}
    for name, metric in metrics.items():
        metric.attach(evaluator, name)
    Loss(criterion, output_transform=lambda x: (x[0], x[1]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_data_loader)
        ms = evaluator.state.metrics
        logger.info("Validation Results - Epoch: [{}/{}]  Avg accuracy: {:.6f} Avg loss: {:.6f}"
              .format(trainer.state.epoch, trainer.state.max_epochs, ms['accuracy'], ms['nll']))


    '''======================early stopping =========================='''
    # def score_function(engine):
    #     val_loss = engine.state.metrics['nll']
    #     return -val_loss
    # handler = EarlyStopping(patience=5, score_function=score_function, trainer=trainer)
    # evaluator.add_event_handler(Events.COMPLETED, handler)

    '''==================print information by iterator========================='''


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        if trainer.state.iteration % args.log_step == 0:
            logger.info("Epoch[{}/{}] Step[{}/{}] Loss: {:.6f}".format(trainer.state.epoch,
                                                                       trainer.state.max_epochs,
                                                                       trainer.state.iteration % steps,
                                                                       steps,
                                                                       trainer.state.output * args.gradient_accumulation_steps)
                        )
    '''================add check point========================'''
    checkpoint_handler = ModelCheckpoint(checkpoint_dir, 'checkpoint', n_saved=3)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'BertClassificationModel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

    '''==============run trainer============================='''
    trainer.run(train_data_loader, max_epochs=args.n_epochs)


if __name__ == '__main__':


    train()
