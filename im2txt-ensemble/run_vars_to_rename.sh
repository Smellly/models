path='ensemble/model/8/'
NEW_CHECKPOINT_FILE=${path}"new_tf_version/model.ckpt-998142"
OLD_CHECKPOINT_FILE=${path}"model.ckpt-998142"

export CUDA_VISIBLE_DEVICES=3
python vars_to_rename.py $path $OLD_CHECKPOINT_FILE $NEW_CHECKPOINT_FILE

