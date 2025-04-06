# JAX-Diffusion

[JAX-Diffusion on Hugging Face 🤗](https://huggingface.co/spaces/carrycooldude/jax-diffusion)

JAX-Diffusion is a project that implements diffusion models using **JAX**, a high-performance numerical computing library. Diffusion models are a class of generative models that have gained popularity for their ability to generate high-quality data samples.

🚀 **Live Demo**: [Try it on Hugging Face Spaces](https://huggingface.co/spaces/carrycooldude/jax-diffusion)

---
![Screenshot from 2025-04-06 14-32-42](https://github.com/user-attachments/assets/c323f134-4a51-4c50-9a5b-1d99065bc849)


---

## Features

- Implementation of diffusion models in **JAX**.
- High-performance and scalable computations.
- Modular and extensible codebase.
- Interactive Gradio app for easy experimentation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/JAX-Diffusion.git
   cd JAX-Diffusion
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Train a diffusion model:
```bash
python train.py --config configs/default.yaml
```

### Generate samples:
```bash
python generate.py --model checkpoints/model.pth
```

### Run the Gradio App:
```bash
python app.py
```

This will launch a Gradio interface where you can generate samples interactively.

## Gradio App Preview

| Input | Output |
|------|--------|
| Enter a text prompt and set diffusion steps | Generates an image using a simple JAX diffusion model |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [JAX](https://github.com/google/jax) for providing the foundation for numerical computing.
- [Gradio](https://gradio.app/) for making it easy to build interactive demos.
- The research community for advancements in diffusion models.
