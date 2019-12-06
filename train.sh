python keras_mnist.py --epoch 10 --max_value 0
mv trainlog.csv ./train_logs/trainlog_trainnoise00.csv

python keras_mnist.py --epoch 10 --max_value 0.1
mv trainlog.csv ./train_logs/trainlog_trainnoise01.csv

python keras_mnist.py --epoch 10 --max_value 0.3
mv trainlog.csv ./train_logs/trainlog_trainnoise03.csv

python keras_mnist.py --epoch 10 --max_value 0.5
mv trainlog.csv ./train_logs/trainlog_trainnoise05.csv

python keras_mnist.py --epoch 10 --max_value 0.7
mv trainlog.csv ./train_logs/trainlog_trainnoise07.csv
