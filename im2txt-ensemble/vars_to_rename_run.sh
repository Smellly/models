path='ensemble/model/5/'
NEW_CHECKPOINT_FILE=${path}"new_tf_version/model.ckpt-1168803"
OLD_CHECKPOINT_FILE=${path}"model.ckpt-1168803"

export CUDA_VISIBLE_DEVICES=3
python vars_to_rename.py $path $OLD_CHECKPOINT_FILE $NEW_CHECKPOINT_FILE

