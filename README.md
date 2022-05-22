# cmpe597_HW2

#### Part a ####
* vae/vae.ipynb is the script that is used for train/evaluate the vae model
* Jupyter notebook can be run by running all cells in order
* There are pretrained encoder and decoder model in vae folder. The necessary parts for image generation and calculating inception score can be run directly on this trained model. For this, all cells must be run except the cells under the Training markdown. Random vector npy files and the trained models should also be in the same folder as the vae.ipynb.


#### Part b ####
* GAN/GAN.py and WGAN/WGAN.py are training scripts for GAN and WGAN models.
* GAN/GAN_Image_Generation.py and WGAN/WGAN_Image_Generation.py are scripts for image generation and calculating inception score from pretrained GAN and WGAN models. 
 *This part has experimented with generators of different complexity. For example, if you want to do the evaluation using complex GAN generator, you should import the class of that generator and read its file.
```
from GAN_Generator_complex import Generator
G = Generator(g_input_dim = z_dim, g_output_dim = 28*28).to(device)
G = torch.load(f"generator_complex", map_location=torch.device('cuda'))
fake_imgs = plot_random_latent_images(G, "complex")

print("Calculating Inception Score...")
print(inception_score(fake_imgs, cuda=False, batch_size=32, resize=True, splits=10))
```

 *If we will generate images with another generator, the import line and readed file must be updated.
```
from GAN_Generator_middle import Generator
G = Generator(g_input_dim = z_dim, g_output_dim = 28*28).to(device)
G = torch.load(f"generator_middle", map_location=torch.device('cuda'))
fake_imgs = plot_random_latent_images(G, "complex")

print("Calculating Inception Score...")
print(inception_score(fake_imgs, cuda=False, batch_size=32, resize=True, splits=10))
```