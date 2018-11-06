# seq2seq+attention+ocr

## env
python3.6,tensorflow1.11 <br>

## steps
* data generation and data proprecess(group by lengths)
* build vgg16 model for fetures extraction and seq2seq module using *tf.contrib.seq2seq*
* train just by running `python train.py`
* evaluate and predict

## instruction
> use **edit distance** as performance measure;  
> load pretrained [vgg16](https://drive.google.com/file/d/0B4WygwSE8o2PUFZYSEJBOW9hZnM/view) model;  
> some test file to learn dynamic rnn ,attention mechanism;  
> just for a try!



