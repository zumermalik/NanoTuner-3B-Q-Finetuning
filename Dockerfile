# NanoTuner-3B-Q-Finetuning/Dockerfile

# 1. BASE IMAGE: Use a modern NVIDIA CUDA image
# CUDA 12.1 is highly compatible with PyTorch 2.x and the latest bitsandbytes.
# The 'devel' image is safer for fine-tuning as it includes necessary build tools.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# 2. SYSTEM SETUP: Install basic tools and set environment variables
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set environment variables for smooth Python/pip operation
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# 3. NON-ROOT USER SETUP (Hugging Face Best Practice)
# Creates a non-root user and switches to it.
RUN useradd -m -u 1000 user
USER user

# 4. INSTALL PYTHON DEPENDENCIES
# We copy only requirements.txt first to leverage Docker layer caching
COPY --chown=user requirements.txt .
# Install packages from your requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

# 5. COPY APPLICATION CODE
# Copy the entire rest of the repository (train.py, config.yaml, etc.)
COPY --chown=user . .

# 6. DEFAULT COMMAND (Execution)
# This command runs automatically when the Space starts.
# Since you'll use Dev Mode, you'll run 'python train.py' manually, but this is a
# robust default for immediate execution.
CMD ["python", "train.py"]
