
import sys
import tensorflow as tf

def main(path, OLD_CHECKPOINT_FILE, NEW_CHECKPOINT_FILE, old2new = True):
    if old2new:
        vars_to_rename = {
            "lstm/basic_lstm_cell/weights": "lstm/basic_lstm_cell/kernel",
            "lstm/basic_lstm_cell/biases": "lstm/basic_lstm_cell/bias",
        }
    else:
        vars_to_rename = {
            "lstm/basic_lstm_cell/kernel":"lstm/basic_lstm_cell/weights",
            "lstm/basic_lstm_cell/bias" : "lstm/basic_lstm_cell/biases",
                }
    new_checkpoint_vars = {}
    reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
    for old_name in reader.get_variable_to_shape_map():
        if old_name in vars_to_rename:
            new_name = vars_to_rename[old_name]
        else:
            new_name = old_name
        new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(new_checkpoint_vars)

    with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, NEW_CHECKPOINT_FILE)

if __name__ == '__main__':
    assert len(sys.argv) == 4
    print sys.argv
    _, path, OLD_CHECKPOINT_FILE, NEW_CHECKPOINT_FILE = sys.argv
    main(path, OLD_CHECKPOINT_FILE, NEW_CHECKPOINT_FILE )
