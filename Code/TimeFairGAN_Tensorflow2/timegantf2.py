import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tqdm import tqdm


class TimeGAN:
    def __init__(self, args, n_seq, log_dir):
        self.args = args
        self.n_seq = n_seq
        self.seq_len = args.seq_len
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layer
        self.iterations = args.iteration
        self.batch_size = args.batch_size
        self.log_dir = log_dir
        self.command = args.command
        self.n_windows = 0
        self.gamma = 1

        if args.command == 'with_fairness':
            self.S_start_index = args.S_start_index
            self.Y_start_index = args.Y_start_index
            self.w_g_f = args.w_g_f
            self.underpriv_index = args.underpriv_index
            self.desire_index = args.desire_index
            self.priv_index = args.priv_index


        if not log_dir.exists():
            log_dir.mkdir(parents=True)

        self.build_model()

        self.autoencoder_optimizer = Adam()
        self.supervisor_optimizer = Adam()
        self.generator_optimizer = Adam()
        self.discriminator_optimizer = Adam()
        self.embedding_optimizer = Adam()

        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()


    def make_rnn(self, n_layers, hidden_units, output_units, name):
        return Sequential([GRU(units=hidden_units,
                           return_sequences=True,
                           name=f'GRU_{i + 1}') for i in range(n_layers)] +
                      [Dense(units=output_units,
                             activation='sigmoid',
                             name='OUT')], name=name)    

    def build_model(self):
        # Define the TimeGAN components (embedding, generator, supervisor, discriminator)
        self.embedder = self.build_embedder()
        self.recovery = self.build_recovery()
        self.generator = self.build_generator()
        self.supervisor = self.build_supervisor()
        self.discriminator = self.build_discriminator()

    

    def build_embedder(self):
        return self.make_rnn(n_layers=3, 
                    hidden_units=self.hidden_dim, 
                    output_units=self.hidden_dim, 
                    name='Embedder')
    
    def build_recovery(self):
        return self.make_rnn(n_layers=3, 
                    hidden_units=self.hidden_dim, 
                    output_units=self.n_seq, 
                    name='Recovery')

    def build_generator(self):
        return self.make_rnn(n_layers=3, 
                    hidden_units=self.hidden_dim, 
                    output_units=self.hidden_dim, 
                    name='Generator')
        

    def build_supervisor(self):
        return self.make_rnn(n_layers=2, 
                      hidden_units=self.hidden_dim, 
                      output_units=self.hidden_dim, 
                      name='Supervisor')

    def build_discriminator(self):
        return self.make_rnn(n_layers=3, 
                         hidden_units=self.hidden_dim, 
                         output_units=1, 
                         name='Discriminator')
    
    def prepare_dataset(self, data):
        self.n_windows = len(data)
        real_series = (tf.data.Dataset
                       .from_tensor_slices(data)
                       .shuffle(buffer_size=self.n_windows)
                       .batch(self.batch_size))
        real_series_iter = iter(real_series.repeat())
        return real_series_iter
    
    def prepare_random_series(self):
        return iter(tf.data.Dataset
                     .from_generator(self.make_random_data, output_types=tf.float32)
                     .batch(self.batch_size)
                     .repeat())


    def make_random_data(self):
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))

    @tf.function
    def train_autoencoder_init(self, x):
        with tf.GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss_0 = self.args.w_e0 * tf.sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        self.autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)
    
    @tf.function
    def train_supervisor(self, x):
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            g_loss_s = self.mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = self.supervisor.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        self.supervisor_optimizer.apply_gradients(zip(gradients, var_list))
        return g_loss_s


    def get_generator_moment_loss(self, y_true, y_pred):
        y_true_mean, y_true_var = tf.nn.moments(x=tf.cast(y_true, tf.float32), axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=tf.cast(y_pred, tf.float32), axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var


    @tf.function
    def train_generator(self, x, z):
        with tf.GradientTape() as tape:
            y_fake = self.adversarial_supervised(z)
            generator_loss_unsupervised = self.bce(y_true=tf.ones_like(y_fake),
                                            y_pred=y_fake)

            y_fake_e = self.adversarial_emb(z)
            generator_loss_unsupervised_e = self.bce(y_true=tf.ones_like(y_fake_e),
                                                y_pred=y_fake_e)
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_hat = self.synthetic_data(z)
            generator_moment_loss = self.get_generator_moment_loss(x, x_hat)

            if self.command == 'with_fairness':
                G = x_hat[:, self.S_start_index:self.S_start_index + 2]
                I = x_hat[:, self.Y_start_index:self.Y_start_index + 2]
                underpriv_term = tf.reduce_sum(G[:, self.underpriv_index:] * I[:, self.desire_index:]) / tf.reduce_sum(x_hat[:, self.S_start_index + self.underpriv_index:])
                priv_term = tf.reduce_sum(G[:, self.priv_index:] * I[:, self.desire_index:]) / tf.reduce_sum(x_hat[:, self.S_start_index + self.priv_index:])
                generator_fairness_loss = tf.abs(underpriv_term - priv_term)
                generator_loss = (generator_loss_unsupervised +
                                generator_loss_unsupervised_e +
                                self.args.w_g_s * tf.sqrt(generator_loss_supervised) +
                                self.args.w_g_v * generator_moment_loss + 
                                self.w_g_f * generator_fairness_loss)
            else:
                generator_loss = (generator_loss_unsupervised +
                                generator_loss_unsupervised_e +
                                self.args.w_g_s * tf.sqrt(generator_loss_supervised) +
                                self.args.w_g_v * generator_moment_loss)

        var_list = self.generator.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        self.generator_optimizer.apply_gradients(zip(gradients, var_list))
        if self.command == 'with_fairness': 
            return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss, generator_fairness_loss
        else:
            return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @tf.function
    def train_embedder(self, x):
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss = self.args.w_e0 * tf.sqrt(embedding_loss_t0) + self.args.w_es * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        self.embedding_optimizer.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)

    @tf.function
    def get_discriminator_loss(self, x, z):
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self.bce(y_true=tf.ones_like(y_real),
                                    y_pred=y_real)

        y_fake = self.adversarial_supervised(z)
        discriminator_loss_fake = self.bce(y_true=tf.zeros_like(y_fake),
                                    y_pred=y_fake)

        y_fake_e = self.adversarial_emb(z)
        discriminator_loss_fake_e = self.bce(y_true=tf.zeros_like(y_fake_e),
                                        y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @tf.function
    def train_discriminator(self, x, z):
        with tf.GradientTape() as tape:
            discriminator_loss = self.get_discriminator_loss(x, z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        self.discriminator_optimizer.apply_gradients(zip(gradients, var_list))
        return discriminator_loss




    def train(self, data):
        real_series_iter = self.prepare_dataset(data)
        random_series_iter = self.prepare_random_series()

        # Create a summary writer for TensorBoard logging
        writer = tf.summary.create_file_writer(self.log_dir.as_posix())

        X = Input(shape=[self.seq_len, self.n_seq], name='RealData')
        Z = Input(shape=[self.seq_len, self.n_seq], name='RandomData')



        # ------------- Phase 1: Autoencoder Training ------------------
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = Model(inputs=X,
                            outputs=X_tilde,
                            name='Autoencoder')
        
        self.autoencoder.summary()

        plot_model(self.autoencoder,
           to_file=(self.log_dir / 'autoencoder.png').as_posix(),
           show_shapes=True)

        # Training autoencoder
        for iteration in tqdm(range(self.iterations)):
            real_batch = next(real_series_iter)
                      
            # Train the autoencoder
            autoencoder_loss = self.train_autoencoder_init(real_batch)

            # Log the autoencoder loss
            with writer.as_default():
                tf.summary.scalar("Autoencoder Loss", autoencoder_loss, step=iteration)

        self.autoencoder.save(self.log_dir / 'autoencoder.keras')

        # ------------- Phase 2: Supervised training ------------------
        for iteration in tqdm(range(self.iterations)):
            real_batch = next(real_series_iter)
            step_g_loss_s = self.train_supervisor(real_batch)
            with writer.as_default():
                tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=iteration)
        
        self.supervisor.save(self.log_dir / 'supervisor.keras')

        # ------------- Phase 3: Joint training ------------------

        # Generator - Adversarial Architecture - Supervised
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        Y_fake =self. discriminator(H_hat)

        self.adversarial_supervised = Model(inputs=Z,
                                    outputs=Y_fake,
                                    name='AdversarialNetSupervised')
        self.adversarial_supervised.summary()
        plot_model(self.adversarial_supervised,
                to_file=(self.log_dir / 'AdversarialNetSupervised.png').as_posix(),
         show_shapes=True)

         # Adversarial Architecture in Latent Space
        Y_fake_e = self.discriminator(E_hat)

        self.adversarial_emb = Model(inputs=Z,
                    outputs=Y_fake_e,
                    name='AdversarialNet')
        self.adversarial_emb.summary()
        plot_model(self.adversarial_emb,
                    to_file=(self.log_dir / 'AdversarialNet.png').as_posix(),
                    show_shapes=True)

        # Mean & Variance Loss
        X_hat = self.recovery(H_hat)
        self.synthetic_data = Model(inputs=Z,
                            outputs=X_hat,
                            name='SyntheticData')
        self.synthetic_data.summary()
        plot_model(self.synthetic_data, show_shapes=True)

        # Discriminator
        # Architecture: Real Data
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=X,
                                    outputs=Y_real,
                                    name='DiscriminatorReal')
        self.discriminator_model.summary()

        


        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_g_loss_fairness =  step_e_loss_t0 = step_d_loss = 0

        for iteration in tqdm(range(self.iterations)):
            
            # Train generator (twice as often as discriminator)
            for kk in range(2):
                X_ = next(real_series_iter)
                Z_ = next(random_series_iter)

                # Train generator
                if self.command == 'with_fairness':
                    step_g_loss_u, step_g_loss_s, step_g_loss_v, step_g_loss_fairness = self.train_generator(X_, Z_)
                else:
                    step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_)
                # Train embedder
                step_e_loss_t0 = self.train_embedder(X_)

            X_ = next(real_series_iter)
            Z_ = next(random_series_iter)
            step_d_loss = self.get_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = self.train_discriminator(X_, Z_)

            if iteration % 1000 == 0:
                if self.command == 'with_fairness':
                    print(f'{iteration:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                        f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | g_loss_fairness: {step_g_loss_fairness:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')
                else:
                    print(f'{iteration:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                        f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')
            with writer.as_default():
                tf.summary.scalar('G Loss S', step_g_loss_s, step=iteration)
                tf.summary.scalar('G Loss U', step_g_loss_u, step=iteration)
                tf.summary.scalar('G Loss V', step_g_loss_v, step=iteration)
                if self.command == 'with_fairness':
                    tf.summary.scalar('G Loss Fair', step_g_loss_fairness, step=iteration)
                tf.summary.scalar('E Loss T0', step_e_loss_t0, step=iteration)
                tf.summary.scalar('D Loss', step_d_loss, step=iteration)

        self.synthetic_data.save(self.log_dir / 'synthetic_data.keras')


    def generate_data(self):
        random_series_iter = self.prepare_random_series()
        generated_data = []
        for i in range(int(self.n_windows / self.batch_size)):
            Z_ = next(random_series_iter)
            d = self.synthetic_data(Z_)
            generated_data.append(d)
        generated_data = np.array(np.vstack(generated_data))
        
        np.save(self.log_dir / 'generated_data.npy', generated_data)
        return generated_data


