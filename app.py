

import os
# Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import gradio as gr
import tensorflow as tf
from tensorflow.keras import models, applications
import numpy as np
import shutil
import warnings
import glob

# Hide specific Keras UserWarning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Skipping variable loading for optimizer.*")

# --- 2. FILE & DIRECTORY SETUP ---
# (File copying code is removed, as it's no longer needed)

# --- 3. CONFIGURATION ---
MODEL_PATH = 'MODEL_PATH = 'best_galaxy_classifier.keras'
CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", 
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral", 
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge"
]

# --- 4. GALAXY INFORMATION DATABASE ---
GALAXY_DESCRIPTIONS = {
    "Disturbed": (
        "**What It Is:** A galaxy that is being gravitationally pulled on by another nearby galaxy.  \n"
        "**What to Look For:** These galaxies look 'messy' or asymmetrical. Key features include:  \n"
        "&nbsp;&nbsp;&nbsp;• **Tidal Tails:** Long, faint streamers of stars and gas being pulled away.  \n"
        "&nbsp;&nbsp;&nbsp;• **Warped Disks:** The galaxy's flat disk appears bent or twisted."
    ),
    "Merging": (
        "**What It Is:** The final, chaotic stage of a collision between two or more galaxies.  \n"
        "**What to Look For:** This is a step beyond 'Disturbed.' You can often see:  \n"
        "&nbsp;&nbsp;&nbsp;• **Two Nuclei (Cores):** The bright centers of the original galaxies are still visible.  \n"
        "&nbsp;&nbsp;&nbsp;• **Highly Irregular Shape:** The entire system is a jumble of stars, dust, and gas."
    ),
    "Round Smooth": (
        "**What It Is:** A classic 'Elliptical' galaxy (Type E0-E2). These are often giant, ancient galaxies.  \n"
        "**What to Look For:** A smooth, featureless, circular ball of light.  \n"
        "&nbsp;&nbsp;&nbsp;• **No Spiral Arms:** Absolutely no spiral structure is visible.  \n"
        "&nbsp;&nbsp;&nbsp;• **Fuzzy Edges:** It smoothly fades into the blackness of space."
    ),
    "In-between Round Smooth": (
        "**What It Is:** An Elliptical or Lenticular galaxy that is noticeably oval-shaped (Type E3-E7).  \n"
        "**What to Look For:** It's in-between a round galaxy and a flat cigar shape.  \n"
        "&nbsp;&nbsp;&nbsp;• **Smooth & Featureless:** No spiral arms or dust lanes.  \n"
        "&nbsp;&nbsp;&nbsp;• **Elongated Shape:** It's clearly a 'squashed' circle."
    ),
    "Cigar Shaped Smooth": (
        "**What It Is:** A highly elongated Elliptical or 'S0' galaxy.  \n"
        "**What to Look For:** A smooth, featureless streak of light.  \n"
        "&nbsp;&nbsp;&nbsp;• **Very Elongated:** Looks like a cigar or an American football.  \n"
        "&nbsp;&nbsp;&nbsp;• **No Spiral Arms:** Completely smooth texture."
    ),
    "Barred Spiral": (
        "**What It Is:** A spiral galaxy, similar to our own Milky Way, that has a distinct bar of stars across its center.  \n"
        "**What to Look For:** A bright, straight bar of stars cutting through the galaxy's nucleus.  \n"
        "&nbsp;&nbsp;&nbsp;• **Arms from Bar:** The spiral arms will begin at the *ends* of the bar, not at the center."
    ),
    "Unbarred Tight Spiral": (
        "**What It Is:** A spiral galaxy (Type Sa-Sb) with no central bar.  \n"
        "**What to Look For:** The spiral arms originate directly from the large, bright central bulge.  \n"
        "&nbsp;&nbsp;&nbsp;• **Tightly Wound:** The arms are wrapped very closely around the galaxy's center."
    ),
    "Unbarred Loose Spiral": (
        "**What It Is:** A spiral galaxy (Type Sc-Sd) with no central bar and a smaller bulge.  \n"
        "**What to Look For:** The spiral arms are prominent, open, and clearly separated.  \n"
        "&nbsp;&nbsp;&nbsp;• **Loosely Wound:** The arms are 'fluffy' and spread far out from the center."
    ),
    "Edge-on without Bulge": (
        "**What It Is:** A flat, disk-shaped galaxy viewed perfectly from its side, with no significant central bulge.  \n"
        "**What to Look For:** A thin, straight line of stars.  \n"
        "&nbsp;&nbsp;&nbsp;• **No Bright Center:** Lacks a bright, round bulge at its midpoint. Often looks like a needle."
    ),
    "Edge-on with Bulge": (
        "**What It Is:** A disk galaxy (like a spiral) viewed from its side, which *does* have a bright central bulge.  \n"
        "**What to Look For:** A 'flying saucer' or 'Sombrero' shape.  \n"
        "&nbsp;&nbsp;&nbsp;• **Bright, Round Center:** A clear, bright bulge sticks out from the flat disk.  \n"
        "&nbsp;&nbsp;&nbsp;• **Dust Lane:** Often has a dark line of dust cutting through the middle."
    )
}

GALAXY_FACTS = {
    "Disturbed": "**Fact:** The 'tidal tails' seen in disturbed galaxies are stars and gas pulled out by gravity, like a gravitational tug-of-war between two galaxies.",
    "Merging": "**Fact:** Galaxy mergers, which take hundreds of millions of years, often trigger a massive burst of star formation called a 'starburst' and can ignite the supermassive black holes at their centers, creating a quasar.",
    "Round Smooth": "**Fact:** These 'elliptical' galaxies are often called 'red and dead.' They are composed mostly of very old, red stars and contain very little gas or dust, so new star formation has almost completely stopped.",
    "In-between Round Smooth": "**Fact:** These galaxies, often called 'lenticular' (lens-shaped), are considered a bridge between spirals and ellipticals. They have a central bulge and a disk, but have lost most of their gas and thus have no spiral arms.",
    "Cigar Shaped Smooth": "**Fact:** While it looks like a 2D cigar, this is a 3D 'ellipsoid' (like a football) viewed from the side. These are some of the most massive, and most rare, types of galaxies in the universe.",
    "Barred Spiral": "**Fact:** Our own Milky Way is a barred spiral galaxy! The central bar is thought to act like a funnel, channeling gas and dust toward the central black hole and triggering star formation.",
    "Unbarred Tight Spiral": "**Fact:** The 'tightness' of the spiral arms is linked to the mass of the central bulge. Galaxies with very large, massive bulges tend to have the most tightly wound arms.",
    "Unbarred Loose Spiral": "**Fact:** Loosely wound spiral arms are often sites of vigorous new star formation. The blue, bright knots you can see in the arms are massive clusters of young, hot stars.",
    "Edge-on without Bulge": "**Fact:** A disk galaxy with no central bulge (or a very small one) is often a 'late-type' spiral (like a Sc or Sd). This means it's likely very rich in gas and is actively forming many new stars.",
    "Edge-on with Bulge": "**Fact:** The dark line you see bisecting the galaxy is a 'dust lane' within the galaxy's disk, composed of interstellar dust that absorbs the light from the stars behind it."
}

GALAXY_LINKS = {
    "Disturbed": "https.astronomy.swin.edu.au/cosmos/D/disturbed+galaxies",
    "Merging": "https://science.nasa.gov/mission/hubble/science/science-highlights/galaxy-details-and-mergers/",
    "Round Smooth": "https://science.nasa.gov/universe/galaxies/elliptical-galaxies/",
    "In-between Round Smooth": "https://www.nasa.gov/universe/galaxies/lenticular-galaxies/",
    "Cigar Shaped Smooth": "https://science.nasa.gov/universe/galaxies/elliptical-galaxies/",
    "Barred Spiral": "https://science.nasa.gov/missions/hubble/hubble-filters-a-barred-spiral/",
    "Unbarred Tight Spiral": "https://esahubble.org/images/potw2020a/",
    "Unbarred Loose Spiral": "https://esahubble.org/images/potw2104a/",
    "Edge-on without Bulge": "https://esahubble.org/images/opo0107a/",
    "Edge-on with Bulge": "https://esahubble.org/images/potw2404a/"
}

GALAXY_RARITY = {
    "Disturbed": "Common (1,081 images, 6.1%)",
    "Merging": "Common (1,853 images, 10.4%)",
    "Round Smooth": "Very Common (2,645 images, 14.9%)",
    "In-between Round Smooth": "Common (2,027 images, 11.4%)",
    "Cigar Shaped Smooth": "Very Rare (334 images, 1.9%)",
    "Barred Spiral": "Common (2,043 images, 11.5%)",
    "Unbarred Tight Spiral": "Common (1,829 images, 10.3%)",
    "Unbarred Loose Spiral": "Very Common (2,628 images, 14.8%)",
    "Edge-on without Bulge": "Common (1,423 images, 8.0%)",
    "Edge-on with Bulge": "Common (1,873 images, 10.6%)"
}

# --- 5. LOAD THE TRAINED MODEL ---
print("--> Loading the trained Keras model...")
model = models.load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# --- 6. DEFINE THE PREDICTION FUNCTION ---
def predict_galaxy(input_image: np.ndarray):
    """
    Takes an image, preprocesses it, and returns:
    1. A dictionary of class confidences.
    2. A markdown string with the rarity.
    3. A markdown string with the description.
    4. A markdown string with a fact.
    5. A markdown string with a link.
    6. A Gradio update to make the profile section visible.
    """
    img_resized = tf.image.resize(input_image, [256, 256])
    img_batch = tf.expand_dims(img_resized, 0)
    img_preprocessed = applications.efficientnet_v2.preprocess_input(img_batch)
    
    prediction_probs = model.predict(img_preprocessed, verbose=0)[0]
    confidences = {CLASS_NAMES[i]: float(prediction_probs[i]) for i in range(len(CLASS_NAMES))}
    
    top_class_index = np.argmax(prediction_probs)
    top_class_name = CLASS_NAMES[top_class_index]
    
    rarity_text = f"**Rarity in Galaxy10 Dataset:** {GALAXY_RARITY[top_class_name]}"
    top_description = GALAXY_DESCRIPTIONS[top_class_name]
    top_fact = GALAXY_FACTS[top_class_name]
    link_url = GALAXY_LINKS[top_class_name]
    explore_link_md = f"➡️ **Explore more:** [Click here to read about {top_class_name} galaxies]({link_url})"
    
    return (
        confidences,
        rarity_text,
        top_description,
        top_fact,
        explore_link_md,
        gr.update(visible=True) # Command to make the group visible
    )

# --- 7. CREATE AND LAUNCH THE GRADIO INTERFACE ---
print("\n--> Launching Gradio Interface...")

banner_image_path = "/kaggle/input/example/Andromeda_Galax.jpg"
banner_types_url = "/kaggle/input/types-of-images/type_of_galaxies.jpg" 

dark_theme_css = """
body { background-color: #0B0F19; color: white; }
.gradio-container { font-size: 1.4rem !important; }
h1, h2, h3 { font-size: 115% !important; font-weight: 600 !important; }
.gr-tab-item { font-size: 1.2rem !important; font-weight: 500 !important; }
.gr-button { font-size: 1.2rem !important; }
.gr-tabs .gr-markdown p { line-height: 1.6 !important; }
.gr-markdown[data-testid="output_rarity"] p { font-size: 1.15rem !important; line-height: 1.6 !important; }
.gr-label-value { font-size: 1.2rem !important; }
"""

with gr.Blocks(theme=gr.themes.Glass(), css=dark_theme_css) as demo:
    # --- The Header (Static Banners) ---
    gr.Markdown("# 🌌 Galaxy Morphology Classifier")

    # #############################################################
    # ##  UPDATED: Reverted banner to its original state         ##
    # #############################################################
    gr.Image(
        value=banner_image_path, 
        label="The Andromeda Galaxy (M31)", 
        interactive=False, 
        height=400
    )
    
    gr.Markdown("# 🌠 Types of Galaxies in Our Model")
    gr.Image(
        value=banner_types_url, 
        label="Classification Types", 
        interactive=False,
        height=350
    )
    
    # --- Use Cases Section ---
    with gr.Accordion("🔭 Real-World Use Cases", open=False):
        gr.Markdown(
            """
            This tool is a prototype for real-world astronomical research. Its primary use cases are:
            * **Automated Data Triaging:** Instantly classifying millions of new galaxy images from sky surveys to find interesting targets.
            * **Target Selection:** Helping researchers find large populations of rare objects, like 'Merging' or 'Disturbed' galaxies, for specific studies.
            * **Citizen Science Augmentation:** Acting as a "first pass" classifier to help focus the efforts of human volunteers on the most ambiguous or scientifically interesting images.
            """
        )
    
    gr.Markdown("--- \n Upload your own image below to classify its morphological type. This prototype is powered by a fine-tuned EfficientNetV2 model that achieved 82% accuracy.")

    # --- The Interactive Classifier ---
    with gr.Row(variant="panel"):
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Galaxy Image")
            submit_btn = gr.Button("Classify", variant="primary")
            
        with gr.Column(scale=2):
            output_label = gr.Label(num_top_classes=5, label="Classification Results")

    # --- Full-width "Pop-Out" Profile Section (Hidden) ---
    with gr.Group(visible=False) as profile_section:
        gr.Markdown("---")
        gr.Markdown("## 🔬 Classified Galaxy Profile")
        with gr.Row():
            with gr.Column(scale=2):
                output_rarity = gr.Markdown(label="Class Rarity")
                with gr.Tabs():
                    with gr.TabItem("Description"):
                        output_description = gr.Markdown()
                    with gr.TabItem("Interesting Fact"):
                        output_fact = gr.Markdown()
                    with gr.TabItem("Learn More"):
                        output_link = gr.Markdown()
            
    # --- Connect Events to Functions ---
    outputs_list = [
        output_label, 
        output_rarity,
        output_description, 
        output_fact, 
        output_link, 
        profile_section
    ]

    submit_btn.click(
        fn=predict_galaxy, 
        inputs=input_image, 
        outputs=outputs_list
    )

# Launch the app
demo.launch(share=True, debug=True)

