# Interactive Artistic Visual Storytelling with Gemini & Stable Diffusion

This project implements an interactive visual storytelling tool within a Kaggle Notebook environment. Users provide an initial story idea, and the system iteratively generates story segments using Google's Gemini 1.5 Flash model. For each segment, it then generates a corresponding image in a user-selected Indian art style using a locally run Stable Diffusion (2.1-base) model. The user can guide the story's progression, regenerate parts, and finally view/save a complete HTML storyboard. The system also includes an automated evaluation step to measure text-image alignment (CLIP similarity) and story coherence (Sentence-BERT).

## Features

*   **Interactive Story Generation:** Users guide the narrative by providing initial ideas and subsequent prompts.
*   **AI-Powered Text Generation:** Leverages Google Gemini 1.5 Flash for creative and coherent story segments.
*   **AI-Powered Image Generation:** Uses a local Stable Diffusion (stabilityai/stable-diffusion-2-1-base) model for generating artistic visuals.
*   **Art Style Specialization:** Focuses on various traditional Indian art styles (e.g., Madhubani, Warli, Kalamkari). Gemini is prompted to generate image prompts specifically tailored to the chosen art style.
*   **Regeneration Capabilities:** Users can opt to regenerate the last story segment or the last generated image.
*   **Storyboard Output:** Generates an HTML file displaying the story text and images in a grid format.
*   **Automated Evaluation:**
    *   Calculates CLIP similarity between generated images and their corresponding (a) detailed image prompts and (b) story text.
    *   Calculates text coherence between consecutive story segments using Sentence-BERT.
*   **Kaggle Environment Optimized:** Designed to run in Kaggle Notebooks, utilizing Kaggle Secrets for API keys and GPU resources.
*   **Directory Management:** Includes initial cleanup of the `/kaggle/working/` output directory.

## Technologies Used

*   **Language Model (LLM):** Google Gemini 1.5 Flash (via `google-generativeai`)
*   **Image Generation Model:** Stable Diffusion 2.1-base (via `diffusers`, `transformers`, `accelerate`)
*   **Programming Language:** Python
*   **Core Libraries:**
    *   `torch`: For deep learning operations and GPU management.
    *   `Pillow`: For image manipulation.
    *   `huggingface_hub`: For model downloads and Hugging Face login.
    *   `kaggle_secrets`: For securely accessing API keys in Kaggle.
    *   `IPython.display`: For rich output in Jupyter/Colab environments.
    *   `sentence-transformers`: For calculating text coherence.
    *   `transformers` (CLIP): For calculating image-text similarity.
*   **Environment:** Kaggle Notebooks (GPU recommended: T4 x2, P100, or similar).

## Prerequisites

1.  **Kaggle Account:** To run the notebook in the Kaggle environment.
2.  **GPU Enabled:** The notebook requires a GPU for Stable Diffusion image generation. Ensure a GPU accelerator (e.g., T4 x2) is selected in your Kaggle Notebook settings.
3.  **Kaggle Secrets:**
    *   `GOOGLE_API_KEY`: Your API key for Google AI Studio (Gemini).
    *   `HUGGINGFACE_TOKEN`: Your Hugging Face user access token (read access is sufficient for downloading models). This is used to authenticate with the Hugging Face Hub.
    You need to add these as "Secrets" in your Kaggle Notebook editor (Add-ons -> Secrets).
4.  **Internet Connection:** Required for downloading libraries, models, and API communication.

## File Structure

The script operates primarily within the Kaggle environment:

*   `interactive_artistic_visual_storytelling.py`: The main Python script (derived from a Colab/Jupyter Notebook).
*   `/kaggle/working/`: Default output directory for Kaggle.
    *   `story_images/`: Directory created to store generated images (e.g., `scene_0.png`, `scene_1_regen_169...png`).
    *   `storyboard_<art_style>.html`: The final HTML storyboard file.

## How to Run

The script is structured as a series of cells, intended to be run sequentially in a Jupyter/Colab-like environment (Kaggle Notebooks).

1.  **Initial Directory Cleanup (Optional but Recommended):**
    *   The script begins with code to clean the `/kaggle/working/` directory. This is useful for fresh runs.

2.  **Hugging Face Login & GPU Check:**
    *   Installs necessary libraries (`diffusers`, `transformers`, etc.).
    *   Logs into Hugging Face Hub using the `HUGGINGFACE_TOKEN` secret.
    *   Verifies GPU availability.

3.  **CELL 1: Setup**
    *   Installs/upgrades main libraries (`google-generativeai`, `diffusers`, etc.).
    *   Imports all necessary modules.
    *   Configures the Google AI Studio API key using the `GOOGLE_API_KEY` secret.
    *   Defines safety settings for Gemini.
    *   Checks for GPU availability again.
    *   Initializes the Gemini model.
    *   Loads the Stable Diffusion pipeline (`stabilityai/stable-diffusion-2-1-base`) onto the GPU if available.
    *   Creates the `story_images` output directory.

4.  **CELL 2: Core Generation Logic**
    *   Defines generation configurations for Gemini (for story and image prompts).
    *   `gemini_generate_story_segment()`: Function to generate a short story paragraph based on history.
    *   `gemini_generate_detailed_image_prompt()`: Function to generate a detailed, art-style-specific image prompt from a story segment.
    *   `generate_visual_asset()`: Function to generate an image using the local Stable Diffusion pipeline based on a prompt and art style, then saves it.
    *   **Important:** This cell must be run to define these core functions before proceeding to Cell 3.

5.  **CELL 3: Interactive Storytelling Loop**
    *   This is the main interactive part of the script.
    *   **Art Style Selection:** Prompts the user to choose an Indian art style from a predefined list.
    *   **Initial Idea:** Prompts the user for the starting idea of the story.
    *   **Interactive Loop (up to `max_turns`):**
        1.  **Story Generation:** Generates the next story segment using Gemini.
        2.  **Prompt Generation:** Generates a detailed image prompt for the story segment, tailored to the chosen art style.
        3.  **Image Generation:** Generates an image using Stable Diffusion (if GPU is available).
        4.  **Display:** Shows the draft story text and generated image.
        5.  **User Action:** Prompts the user for the next step:
            *   `[C]ontinue`: Accepts the current scene and prompts for input for the next scene.
            *   `[L]LM Continue`: Accepts the current scene and lets the LLM continue the story naturally.
            *   `Re[G]en Story`: Regenerates the story segment (and subsequently the prompt and image).
            *   `Re[I]m Image`: Regenerates only the image using the existing story segment and detailed prompt.
            *   `[E]nd`: Finishes the story. The currently generated (but not yet accepted) scene will be added if valid.
    *   **Storyboard Generation:** After the loop ends (or `max_turns` is reached), it compiles all accepted scenes into an HTML file (`storyboard_<art_style>.html`) with images embedded (as base64) and text, displayed in a grid. This file is saved to `/kaggle/working/`.

6.  **CELL 4: Quantitative Evaluation**
    *   This cell should be run *after* Cell 3 has completed and a storyboard with at least one scene (preferably more) has been generated.
    *   **CLIP Similarity:**
        *   Loads a CLIP model (`openai/clip-vit-base-patch32`).
        *   Calculates and prints the average similarity score between:
            *   The detailed image prompts and their generated images.
            *   The story text segments and their generated images.
        *   Unloads the CLIP model to free GPU memory.
    *   **Text Coherence:**
        *   Loads a Sentence-BERT model (`all-MiniLM-L6-v2`).
        *   Calculates and prints the average cosine similarity between embeddings of consecutive story segments.
    *   **Summary:** Prints a final summary of the calculated metrics.

## Key Script Components

*   **Directory Cleaning:** Ensures a clean slate in `/kaggle/working/` by removing previous files and folders.
*   **API/Model Initialization (Cell 1):** Handles setup of Gemini and Stable Diffusion models, including API key configuration and Hugging Face login.
*   **Story Generation (`gemini_generate_story_segment` in Cell 2):** Takes story history and generates the next coherent paragraph.
*   **Image Prompt Generation (`gemini_generate_detailed_image_prompt` in Cell 2):** Translates a story segment into a rich, art-style-specific prompt for Stable Diffusion, emphasizing artistic keywords and avoiding photorealism.
*   **Image Generation (`generate_visual_asset` in Cell 2):** Uses the local Stable Diffusion pipeline to create an image from the detailed prompt.
*   **Interactive Loop (Cell 3):** Manages user interaction, orchestrates generation steps, handles regeneration requests, and builds the `storyboard_data` list.
*   **Storyboard Output (Cell 3):** Formats and saves the final story as an HTML file for easy viewing.
*   **Evaluation Metrics (Cell 4):** Provides quantitative measures of the system's output quality using CLIP and Sentence-BERT.

## Troubleshooting & Notes

*   **GPU Out of Memory:** If you encounter `torch.cuda.OutOfMemoryError`, especially during image generation or CLIP model loading:
    *   The script attempts to clear the CUDA cache (`torch.cuda.empty_cache()`) in some error handlers.
    *   Restarting the Kaggle kernel and running cells sequentially might help.
    *   Ensure no other notebooks are consuming GPU resources on your Kaggle instance.
    *   The `stabilityai/stable-diffusion-2-1-base` model is moderately sized. If issues persist, a smaller model variant might be needed (requiring code changes), or a more powerful GPU.
*   **API Key Errors:** Double-check that `GOOGLE_API_KEY` and `HUGGINGFACE_TOKEN` are correctly set up in Kaggle Secrets and that the keys themselves are valid.
*   **Model Download Times:** The first run of Cell 1 (loading SD) and Cell 4 (loading CLIP/Sentence-BERT) might take several minutes as models are downloaded.
*   **Safety Blocks:** Gemini has safety filters. If story or prompt generation is blocked, the script will indicate this and may use fallbacks or fail the step.
*   **Running Cells Out of Order:** Running cells out of the intended sequence (e.g., Cell 3 before Cell 2) will lead to errors as functions or variables will not be defined.

## Potential Future Enhancements

*   Integration of more diverse art styles.
*   Option to use different Stable Diffusion checkpoints or other image generation models.
*   More sophisticated story planning or plot control mechanisms.
*   User ability to upload a reference image for style transfer.
*   Saving and loading story progress.
*   A more interactive UI (e.g., using Gradio or Streamlit, if deployed outside Kaggle notebooks).
*   Allowing choice of different LLM models.
