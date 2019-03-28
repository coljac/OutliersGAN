import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 1000
    checkpoint_dir = '/fred/oz012/Colin/bruno/checkpoints/' + str(FLAGS.checkpoint) + '/'
    tensorboard_dir = '/fred/oz012/Colin/bruno/tensorboard/' + str(FLAGS.checkpoint) + '/'

    with tf.Session() as sess:
        dcgan = DCGAN(sess=sess,
                      in_height=3500,
                      in_width=1,
                      nchannels=1,
                      batch_size=512,
                      noise_dim=100,
                      mode='wgan-gp', # w = wasserstein gradient penalty
                      opt_pars=(0.00005, 0.0, 0.9), # optimizer LR, momentum
                      d_iters=5, # discriminator its per gen it (1:1 non Wasserstein)
                      data_name='spectra', # e.g. mnist, f-mnist, cifar10, spectra
                      #dataset_size=345960,
                      dataset_size=22791, #legacy_bit6, 315082, # Invalid: 391672
                      # Galaxies: 321520 (star forming and starburst, with zWarning)
                      pics_save_names=('qso_now_spec_data_','qso_now_spec_gen_'),
                      #files_path='/fred/oz012/Bruno/data/spectra/boss/loz/',
                      # files_path='/fred/oz012/Bruno/data/spectra/legacy/legacy_bit6/', # Data in TF format
                      files_path='/fred/oz012/Bruno/data/spectra/qso_zWarning/', # Data in TF format
                      # files_name='spectra_grid', # ... times N shards
                      files_name='spectra', # ... times N shards
                      checkpoint_dir=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir)

        if FLAGS.mode == 'train':
            dcgan.train(nepochs, drop_d=0.0, drop_g=0.0, flip_prob=0.15, restore=False)

        elif FLAGS.mode == 'generate':
            dcgan.generate(N=1, n=5, name='generate', write_fits=True) # N pic, n spectra

        elif FLAGS.mode == 'predict':
            dcgan.predict(n_pred=514)
        
        elif FLAGS.mode == 'save_features': # Store model (for PCA etc)
            inp = 315082
            dcgan.save_features(ninputs=inp, 
                                # Where to save the layer
                                save_path='/fred/oz012/Bruno/data/hdf5/tf_legacy_loz_'+str(inp)+'.hdf5', 
                        additional_data_name='spectra', # additional data type raw data
                        additional_data_path='/fred/oz012/Bruno/data/spectra/boss/loz/',
                        #additional_data_path='/fred/oz012/Bruno/data/spectra/legacy/outliers/',
                        additional_ninputs=inp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
