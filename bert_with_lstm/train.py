# coding=utf-8
import datetime
import shutil

import tensorflow as tf

from bert_with_lstm.BiLSTM import BiLSTMWithAttention
from bert_with_lstm.metrics import *
from bert_with_lstm.dataset import *
import os

tf.logging.set_verbosity(tf.logging.INFO)

labelList = data.getLabelList()

train_example = data.get_train_input_example()

eval_example = data.get_eval_input_example()

labelListRange = range(len(labelList))

init_model_in_dir = True

# 判断目录是否为空
if not os.listdir(config.savedModelPathForCkpt):
    init_model_in_dir = False

print("init ==> ", init_model_in_dir)

eval_max_acc = 0

# 定义计算图
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
        #
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        if init_model_in_dir:

            checkpoint_file = tf.train.latest_checkpoint(config.savedModelPathForCkpt)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            print(tf.global_variables_initializer())

            print('--------------------trainable variables---------------------------')

            # 查看模型中的trainable variables
            tvs = [v for v in tf.trainable_variables()]
            for v in tvs:
                print(v.name)
                print(sess.run(v))
            print('--------------------tensor或者operations---------------------------')
            # 查看模型中的所有tensor或者operations
            gv = [v for v in tf.global_variables()]
            for v in gv:
                print(v.name)
            print('--------------------operations相关的tensor---------------------------')
            # 获得几乎所有的operations相关的tensor
            ops = [o for o in sess.graph.get_operations()]
            for o in ops:
                print(o.name)
            print('-----------------------------------------------')

            globalStep = graph.get_tensor_by_name("globalStep:0")
            # globalStep = tf.Variable(0, name="globalStep", trainable=False)

            inputX = graph.get_operation_by_name("inputX").outputs[0]
            inputY = graph.get_operation_by_name("inputY").outputs[0]
            dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]

            _loss = graph.get_tensor_by_name("loss/add:0")
            _predictions = graph.get_tensor_by_name("output/predictions:0")

            summaryOp = graph.get_tensor_by_name("Merge/MergeSummary:0")

            # trainOp = graph.get_tensor_by_name("bi-lstm0/fw/lstm_cell/kernel/Adam:0")

            trainOp = graph.get_operation_by_name("Adam").outputs[0]

            # # 定义优化函数，传入学习速率参数
            # optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # # 计算梯度,得到梯度和变量
            # gradsAndVars = optimizer.compute_gradients(_loss)
            # # 将梯度应用到变量下，生成训练器
            # trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        else:

            lstm = BiLSTMWithAttention(config)

            inputX = lstm.inputX
            inputY = lstm.inputY
            dropoutKeepProb = lstm.dropoutKeepProb
            _loss = lstm.loss
            _predictions = lstm.predictions

            globalStep = tf.Variable(0, name="globalStep", trainable=False)

            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # 计算梯度,得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(lstm.loss)
            # 将梯度应用到变量下，生成训练器
            trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

            # 用summary绘制tensorBoard
            gradSummaries = []
            for g, v in gradsAndVars:
                if g is not None:
                    tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

            lossSummary = tf.summary.scalar("loss", lstm.loss)
            summaryOp = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        if os.path.exists(config.savedModelPathForPb):
            shutil.rmtree(config.savedModelPathForPb)
        builder = tf.saved_model.builder.SavedModelBuilder(config.savedModelPathForPb)



        print("trainOp ==> ", trainOp, " summaryOp ==> ", summaryOp, " _loss ==> ", _loss, " globalStep ==> ", globalStep, " _predictions ==> ", _predictions)

        print(inputX, inputY, dropoutKeepProb)

        def trainStep(batchX, batchY):
            """
            训练函数
            """
            global acc, prec, f_beta, recall
            feed_dict = {
                inputX: batchX,
                inputY: batchY,
                dropoutKeepProb: config.model.dropoutKeepProb
            }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, _loss, _predictions],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()

            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)


            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                              labels=labelListRange)

            trainSummaryWriter.add_summary(summary, step)

            return loss, acc, prec, recall, f_beta


        def devStep(batchX, batchY):
            """
            验证函数
            """
            global acc, precision, f_beta, recall
            feed_dict = {
                inputX: batchX,
                inputY: batchY,
                dropoutKeepProb: 1.0
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, _loss, _predictions],
                feed_dict)

            if config.numClasses == 1:

                acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
            elif config.numClasses > 1:
                acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                                   labels=labelListRange)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, precision, recall, f_beta


        for i in range(int(len(train_example) / config.batchSize * config.training.epoches)):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(train_example, labelList, config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(eval_example, labelList, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    mean_acc = mean(accs)
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,
                                                                                                         currentStep,
                                                                                                         mean(losses),
                                                                                                         mean_acc,
                                                                                                         mean(
                                                                                                             precisions),
                                                                                                         mean(recalls),
                                                                                                         mean(f_betas)))

                    if mean_acc > eval_max_acc:
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, config.savedModelPathForCkpt, global_step=currentStep)
                        eval_max_acc = mean_acc
                        print("Saved model checkpoint to {}\n".format(path))

        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(lstm.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(lstm.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(lstm.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={"predict": prediction_signature},
                                             legacy_init_op=legacy_init_op)

        builder.save()
