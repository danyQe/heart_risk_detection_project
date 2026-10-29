import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
import os
import cv2
from PIL import Image
from skimage import morphology, filters, measure
import google.generativeai as genai
from dotenv import load_dotenv
import io
import base64
from matplotlib.colors import LinearSegmentedColormap
import requests
import tempfile
# Load environment variables
load_dotenv()
import kagglehub

# Download latest version
path = kagglehub.model_download("goutham1208/heart_disease_prediction/other/default")

print("Path to model files:", path)
# Configure Google Generative AI with API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Cardiovascular Risk Assessment System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.2rem;
        color: #1565C0;
        margin-top: 1rem;
    }
    .highlight {
        background-color: #f0f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1E88E5;
    }
    .risk-high {
        color: #D32F2F;
        font-weight: bold;
    }
    .risk-moderate {
        color: #FF8F00;
        font-weight: bold;
    }
    .risk-low {
        color: #388E3C;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #616161;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Custom colormap for vessel visualization
colors = [(1, 1, 1), (0.8, 0, 0)]  # White to red
vessel_cmap = LinearSegmentedColormap.from_list('vessel_cmap', colors, N=256)

# Load models and scalers
# Update the load_models function for better RETFound handling
@st.cache_resource
def load_models():
    """Load all models and scalers needed for prediction"""
    models = {}
    
    # Load vessel segmentation model
    try:
        models['vessel_model'] = load_model(os.path.join(path, 'models/vessel_segmentation_model.keras'))
        st.success("✅ Vessel segmentation model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading vessel segmentation model: {e}")
        models['vessel_model'] = None
    
    # Create lightweight RETFound alternative
    try:
        st.info("Setting up EfficientNet-based feature extractor for retinal images")
        # Create a lightweight deep feature extractor based on EfficientNet
        base_model = tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_shape=(512, 512, 3)
        )
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        features = tf.keras.layers.Dense(512, name='retfound_features', activation='relu')(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(features)
        
        retfound_model = tf.keras.Model(inputs=base_model.input, outputs=[features, output])
        # Compile the model to initialize weights
        retfound_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        models['retfound_model'] = retfound_model
        st.success("✅ Deep feature extractor initialized successfully")
    except Exception as e:
        st.error(f"❌ Error creating deep feature extractor: {e}")
        models['retfound_model'] = None
    
    # Load retinal risk model
    try:
        models['retinal_model'] = xgb.XGBClassifier()
        models['retinal_model'].load_model(os.path.join(path, 'models/retinal_cvd_risk_model.json'))
        st.success("✅ Retinal risk model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading retinal risk model: {e}")
        models['retinal_model'] = None
    
    # Load clinical risk model
    try:
        models['clinical_model'] = xgb.XGBClassifier()
        models['clinical_model'].load_model(os.path.join(path, 'models/heart_failure_clinical_model.json'))
        st.success("✅ Clinical risk model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading clinical risk model: {e}")
        models['clinical_model'] = None
    
    # Load scalers
    try:
        models['retinal_scaler'] = joblib.load(os.path.join(path, 'models/feature_scaler.pkl'))
        st.success("✅ Retinal feature scaler loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading retinal feature scaler: {e}")
        models['retinal_scaler'] = None
    
    try:
        models['clinical_scaler'] = joblib.load(os.path.join(path, 'models/clinical_feature_scaler.pkl'))
        st.success("✅ Clinical feature scaler loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading clinical feature scaler: {e}")
        models['clinical_scaler'] = None
    
    # Load Gemini model for text generation
    try:
        models['gemini_model'] = genai.GenerativeModel('gemini-2.0-flash')
        st.success("✅ Gemini model initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing Gemini model: {e}")
        models['gemini_model'] = None
    
    return models
def preprocess_for_retfound(image, target_size=(512, 512)):
    """Preprocess an image for RETFound model"""
    # Resize image if needed
    if image.shape[:2] != target_size:
        image = tf.image.resize(image, target_size, method='lanczos3')
    
    # Normalize using ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    return image

# Image preprocessing functions
def preprocess_image(uploaded_image, target_size=(512, 512)):
    """Preprocess the uploaded image for model input"""
    if isinstance(uploaded_image, bytes):
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(uploaded_image))
    else:
        image = uploaded_image
    
    # Convert to RGB if not already
    image = image.convert('RGB')
    
    # Resize to target size
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    return img_array

def preprocess_for_vessel_segmentation(image):
    """Prepare image for vessel segmentation model"""
    # Extract green channel (best for vessel contrast)
    green_channel = image[:, :, 1]
    
    # Apply CLAHE for better vessel contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply((green_channel * 255).astype(np.uint8)) / 255.0
    
    # Create mask (circular approximation for field of view)
    h, w = image.shape[:2]
    mask = np.ones((h, w))
    center = (h // 2, w // 2)
    radius = min(h, w) // 2 - 10
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mask = dist_from_center <= radius
    mask = mask.astype(np.float32)
    
    # Create a 3-channel input: original green channel, enhanced green channel, and mask
    input_img = np.stack([green_channel, enhanced, mask], axis=-1)
    
    return input_img

def extract_vessel_features(vessel_segmentation, mask=None):
    """Extract quantitative features from vessel segmentation"""
    # If mask is provided, apply it
    if mask is not None:
        vessel_segmentation = vessel_segmentation * mask
    
    # Threshold the vessel segmentation to create a binary vessel map
    # (using Otsu's method for adaptive thresholding)
    vessel_binary = vessel_segmentation > filters.threshold_otsu(vessel_segmentation)
    
    # Calculate vessel density features
    total_pixels = np.sum(mask > 0) if mask is not None else vessel_segmentation.size
    vessel_pixels = np.sum(vessel_binary)
    vessel_density = vessel_pixels / total_pixels
    
    # Skeletonize the vessels to get centerlines
    vessel_skeleton = morphology.skeletonize(vessel_binary)
    
    # Extract vessel tortuosity features
    # 1. Count branch points
    vessel_labels = measure.label(vessel_skeleton)
    props = measure.regionprops(vessel_labels)
    
    # Store features
    features = {
        'vessel_density': vessel_density,
        'vessel_area': vessel_pixels,
        'vessel_count': len(props),
    }
    
    # Calculate vessel width features using distance transform
    if np.sum(vessel_binary) > 0:
        dist_transform = cv2.distanceTransform(vessel_binary.astype(np.uint8), cv2.DIST_L2, 3)
        # Mean vessel width (2 × average distance transform value along the skeleton)
        if np.sum(vessel_skeleton) > 0:
            mean_vessel_width = 2 * np.mean(dist_transform[vessel_skeleton])
            features['mean_vessel_width'] = mean_vessel_width
        
        # Max vessel width
        max_vessel_width = 2 * np.max(dist_transform)
        features['max_vessel_width'] = max_vessel_width
    else:
        features['mean_vessel_width'] = 0
        features['max_vessel_width'] = 0
    
    # Calculate vessel tortuosity measures
    if len(props) > 0:
        # Extract tortuosity features as ratio of skeleton length to Euclidean distance
        tortuosity_values = []
        for prop in props:
            if prop.area > 10:  # Filter out very small segments
                # Calculate Euclidean distance between endpoints
                coords = prop.coords
                if len(coords) > 0:
                    endpoints = []
                    for p in coords:
                        neighbors = 0
                        for dx in [-1, 0, 1]:
                            for dy in [-1, 0, 1]:
                                nx, ny = p[0] + dx, p[1] + dy
                                if 0 <= nx < vessel_skeleton.shape[0] and 0 <= ny < vessel_skeleton.shape[1]:
                                    if vessel_skeleton[nx, ny] and (dx != 0 or dy != 0):
                                        neighbors += 1
                        if neighbors == 1:  # Endpoint has only one neighbor
                            endpoints.append(p)
                    
                    if len(endpoints) >= 2:
                        # Take first and last endpoint
                        euc_dist = np.sqrt((endpoints[0][0] - endpoints[-1][0])**2 + 
                                          (endpoints[0][1] - endpoints[-1][1])**2)
                        if euc_dist > 0:
                            tortuosity = prop.area / euc_dist
                            tortuosity_values.append(tortuosity)
        
        if tortuosity_values:
            features['mean_tortuosity'] = np.mean(tortuosity_values)
            features['max_tortuosity'] = np.max(tortuosity_values)
        else:
            features['mean_tortuosity'] = 0
            features['max_tortuosity'] = 0
    else:
        features['mean_tortuosity'] = 0
        features['max_tortuosity'] = 0
    
    # Calculate quadrant-wise vessel density
    h, w = vessel_segmentation.shape
    center_y, center_x = h // 2, w // 2
    
    # Create quadrant masks
    quadrants = [
        (slice(0, center_y), slice(0, center_x)),  # Top-left
        (slice(0, center_y), slice(center_x, None)),  # Top-right
        (slice(center_y, None), slice(0, center_x)),  # Bottom-left
        (slice(center_y, None), slice(center_x, None))  # Bottom-right
    ]
    
    # Calculate vessel density in each quadrant
    for i, (y_slice, x_slice) in enumerate(quadrants):
        quadrant_mask = np.zeros_like(vessel_segmentation)
        quadrant_mask[y_slice, x_slice] = 1
        
        if mask is not None:
            quadrant_mask = quadrant_mask * mask
        
        quadrant_total = np.sum(quadrant_mask > 0)
        quadrant_vessels = np.sum(vessel_binary[y_slice, x_slice] * quadrant_mask[y_slice, x_slice])
        
        if quadrant_total > 0:
            features[f'quadrant_{i+1}_density'] = quadrant_vessels / quadrant_total
        else:
            features[f'quadrant_{i+1}_density'] = 0
    
    return features, vessel_binary

# Fusion system for integrating retinal and clinical risk assessments
class CardiovascularRiskFusionSystem:
    """
    System for integrating retinal image analysis and clinical data
    for comprehensive cardiovascular risk assessment
    """
    def __init__(self, models):
        """Initialize the fusion system with trained models"""
        self.vessel_model = models['vessel_model']
        self.retinal_model = models['retinal_model']
        self.clinical_model = models['clinical_model']
        self.retinal_scaler = models['retinal_scaler']
        self.clinical_scaler = models['clinical_scaler']
        self.gemini_model = models['gemini_model']
        self.retfound_model=models["retfound_model"]
        
        # Set initial weights (can be optimized based on validation data)
        self.retinal_weight = 0.4
        self.clinical_weight = 0.6
        
        # Get the expected feature count from the retinal scaler
        if self.retinal_scaler is not None:
            self.expected_feature_count = self.retinal_scaler.n_features_in_
        else:
            self.expected_feature_count = 524  # Default value based on our model
    
    def predict_vessel_segmentation(self, image):
        """Predict vessel segmentation"""
        if self.vessel_model is None:
            return None, None
        
        # Preprocess image for vessel segmentation
        input_img = preprocess_for_vessel_segmentation(image)
        input_img = np.expand_dims(input_img, axis=0)
        
        # Predict segmentation
        pred = self.vessel_model.predict(input_img, verbose=0)[0, :, :, 0]
        
        # Create mask
        h, w = image.shape[:2]
        mask = np.ones((h, w))
        center = (h // 2, w // 2)
        radius = min(h, w) // 2 - 10
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask = dist_from_center <= radius
        mask = mask.astype(np.float32)
        
        # Apply mask
        pred_masked = pred * mask
        
        return pred_masked, mask
    
    def extract_retinal_features(self, image):
        """Extract retinal features for risk prediction"""
        # Get vessel segmentation
        vessel_seg, mask = self.predict_vessel_segmentation(image)
        
        # Extract vessel features
        if vessel_seg is not None:
            vessel_features, vessel_binary = extract_vessel_features(vessel_seg, mask)
        else:
            vessel_features = {}
            vessel_binary = None
        
        # Extract RETFound features
        retfound_features = None
        if self.retfound_model is not None:
            # Preprocess for RETFound
            retfound_input = preprocess_for_retfound(image)
            retfound_input = np.expand_dims(retfound_input, axis=0)
            
            # Get RETFound features
            features, prediction = self.retfound_model.predict(retfound_input, verbose=0)
            retfound_features = features[0]  # Extract the feature vector
            retfound_prediction = prediction[0][0]
            
            # Add RETFound prediction to vessel features
            vessel_features['retfound_prediction'] = retfound_prediction
        else:
            # Create dummy features if RETFound model is not available
            retfound_features = np.zeros(512)  # Default size for RETFound features
            retfound_prediction = 0.5
            vessel_features['retfound_prediction'] = retfound_prediction
        
        # Create feature vector combining both feature sets
        feature_vector = []
        for key, value in vessel_features.items():
            if isinstance(value, (int, float)) and key != 'image_path' and key != 'class':
                feature_vector.append(value)
        
        # Combine with RETFound features
        if retfound_features is not None:
            combined_features = np.concatenate([feature_vector, retfound_features])
        else:
            combined_features = np.array(feature_vector)
        
        # Pad or truncate feature vector to match expected dimensions
        current_features = len(combined_features)
        if current_features < self.expected_feature_count:
            # If we have fewer features than expected, pad with zeros
            padding = np.zeros(self.expected_feature_count - current_features)
            combined_features = np.concatenate([combined_features, padding])
            st.warning(f"Warning: Padding feature vector from {current_features} to {self.expected_feature_count} dimensions")
        elif current_features > self.expected_feature_count:
            # If we have more features than expected, truncate
            combined_features = combined_features[:self.expected_feature_count]
            st.warning(f"Warning: Truncating feature vector from {current_features} to {self.expected_feature_count} dimensions")
        
        return combined_features, vessel_seg, vessel_binary
    
    def predict(self, image, clinical_data):
        """
        Make a prediction using both retinal image and clinical data
        """
        try:
            # Extract retinal features
            retinal_features, vessel_seg, vessel_binary = self.extract_retinal_features(image)
            
            if retinal_features is not None and self.retinal_model is not None and self.retinal_scaler is not None:
                # Scale retinal features
                retinal_features_2d = np.array(retinal_features).reshape(1, -1)
                retinal_features_scaled = self.retinal_scaler.transform(retinal_features_2d)
                
                # Get retinal prediction
                retinal_risk = self.retinal_model.predict_proba(retinal_features_scaled)[0, 1]
            else:
                retinal_risk = 0.5  # Default value if feature extraction failed
            
            # Process clinical data
            if self.clinical_model is not None and self.clinical_scaler is not None:
                # Scale clinical data
                clinical_data_scaled = self.clinical_scaler.transform(clinical_data)
                
                # Get clinical prediction
                clinical_risk = self.clinical_model.predict_proba(clinical_data_scaled)[0, 1]
            else:
                clinical_risk = 0.5  # Default value if clinical model failed
            
            # Combine predictions using weighted average
            risk_score = (self.retinal_weight * retinal_risk + 
                        self.clinical_weight * clinical_risk)
            
            return {
                'risk_score': risk_score,
                'retinal_risk': retinal_risk,
                'clinical_risk': clinical_risk,
                'vessel_segmentation': vessel_seg,
                'vessel_binary': vessel_binary
            }
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            # Fallback to default values
            return {
                'risk_score': 0.5,
                'retinal_risk': 0.5,
                'clinical_risk': 0.5,
                'vessel_segmentation': None,
                'vessel_binary': None
            }
    
    def explain_prediction(self, image, clinical_data, prediction_results):
        """
        Explain the prediction by identifying key risk factors
        """
        # Extract vessel features if needed
        vessel_risk_factors = []
        if prediction_results['vessel_segmentation'] is not None:
            vessel_features, _ = extract_vessel_features(
                prediction_results['vessel_segmentation'])
            
            # Check for risk factors
            if vessel_features.get('vessel_density', 1) < 0.55:
                vessel_risk_factors.append(('Low vessel density', vessel_features.get('vessel_density')))
            
            if vessel_features.get('mean_tortuosity', 0) > 3.0:
                vessel_risk_factors.append(('High vessel tortuosity', vessel_features.get('mean_tortuosity')))
            
            # Compare quadrant vessel densities
            quadrant_densities = [
                vessel_features.get('quadrant_1_density', 0),
                vessel_features.get('quadrant_2_density', 0),
                vessel_features.get('quadrant_3_density', 0),
                vessel_features.get('quadrant_4_density', 0)
            ]
            if max(quadrant_densities) - min(quadrant_densities) > 0.15:
                vessel_risk_factors.append(('Uneven vessel distribution', max(quadrant_densities) - min(quadrant_densities)))
        
        # Extract clinical risk factors
        clinical_dict = clinical_data.iloc[0].to_dict()
        clinical_risk_factors = []
        
        # Check key clinical indicators
        if clinical_dict.get('ejection_fraction', 100) < 40:
            clinical_risk_factors.append(('Low ejection fraction', clinical_dict.get('ejection_fraction')))
        
        if clinical_dict.get('serum_creatinine', 0) > 1.3:
            clinical_risk_factors.append(('Elevated serum creatinine', clinical_dict.get('serum_creatinine')))
        
        if clinical_dict.get('serum_sodium', 150) < 135:
            clinical_risk_factors.append(('Low serum sodium', clinical_dict.get('serum_sodium')))
        
        if clinical_dict.get('age', 0) > 65:
            clinical_risk_factors.append(('Advanced age', clinical_dict.get('age')))
        
        # Determine overall risk level
        if prediction_results['risk_score'] > 0.67:
            risk_level = "High"
        elif prediction_results['risk_score'] > 0.33:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            'risk_score': prediction_results['risk_score'],
            'retinal_risk': prediction_results['retinal_risk'],
            'clinical_risk': prediction_results['clinical_risk'],
            'vessel_risk_factors': vessel_risk_factors,
            'clinical_risk_factors': clinical_risk_factors,
            'overall_risk_level': risk_level
        }
    
    def generate_recommendations(self, explanation, clinical_data):
        """Generate personalized recommendations using Gemini"""
        if self.gemini_model is None:
            return "LLM model not available. Unable to generate personalized recommendations."
        
        clinical_dict = clinical_data.iloc[0].to_dict()
        
        # Basic patient profile
        age = clinical_dict.get('age', 'unknown')
        sex = 'Male' if clinical_dict.get('sex', 0) == 1 else 'Female'
        has_diabetes = 'Yes' if clinical_dict.get('diabetes', 0) == 1 else 'No'
        has_hypertension = 'Yes' if clinical_dict.get('high_blood_pressure', 0) == 1 else 'No'
        is_smoker = 'Yes' if clinical_dict.get('smoking', 0) == 1 else 'No'
        ejection_fraction = clinical_dict.get('ejection_fraction', 'unknown')
        
        # Construct prompt
        prompt = f"""You are a cardiovascular health assistant analyzing results from a multi-modal AI system that assessed a patient's cardiovascular risk using retinal imaging and clinical data.

PATIENT PROFILE:
- Age: {age}
- Sex: {sex}
- Diabetes: {has_diabetes}
- Hypertension: {has_hypertension}
- Smoking: {is_smoker}
- Ejection Fraction: {ejection_fraction}%

AI ASSESSMENT RESULTS:
- Overall Risk Score: {explanation['risk_score']:.2f} (Scale: 0-1)
- Overall Risk Level: {explanation['overall_risk_level']}
- Retinal Analysis Risk Score: {explanation['retinal_risk']:.2f}
- Clinical Data Risk Score: {explanation['clinical_risk']:.2f}

KEY RISK FACTORS IDENTIFIED:
"""
        
        # Add retinal risk factors
        if explanation['vessel_risk_factors']:
            prompt += "Retinal Biomarkers:\n"
            for factor, value in explanation['vessel_risk_factors']:
                prompt += f"- {factor}: {value:.2f}\n"
        else:
            prompt += "No significant retinal risk factors identified.\n"
        
        # Add clinical risk factors
        if explanation['clinical_risk_factors']:
            prompt += "\nClinical Indicators:\n"
            for factor, value in explanation['clinical_risk_factors']:
                prompt += f"- {factor}: {value}\n"
        else:
            prompt += "\nNo significant clinical risk factors identified.\n"
        
        # Complete the prompt with instructions
        prompt += f"""
Based on this assessment, please provide:
1. An interpretation of these results in simple, clear language that the patient can understand (2-3 paragraphs)
2. Specific recommendations for next steps (monitoring, lifestyle changes, or medical consultation)
3. A brief explanation of how retinal changes may reflect cardiovascular health
4. A prioritized list of modifiable risk factors the patient should address

Please be empathetic but honest about the level of risk identified ({explanation['overall_risk_level']}), and tailor advice to this {age}-year-old {sex.lower()} patient.
"""
        
        try:
            # Generate response from Gemini
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            return f"Unable to generate recommendations due to an error: {e}"

# Main application interface
def main():
    """Main Streamlit application"""
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    # Create fusion system
    fusion_system = CardiovascularRiskFusionSystem(models)
    
    # App header
    st.markdown('<h1 class="main-header">Cardiovascular Health Assessment System</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="highlight" style="color:black">
        This system analyzes retinal images and clinical data to provide a comprehensive assessment
        of cardiovascular risk. The retina offers a unique window into vascular health, while clinical
        measurements provide complementary information about known risk factors.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Patient Data Entry", "Results & Analysis", "About the System"])
    
    # Tab 1: Patient Data Entry
    with tab1:
        st.markdown('<h2 class="sub-header">Patient Information</h2>', unsafe_allow_html=True)
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        # Column 1: Retinal Image Upload
        with col1:
            st.markdown('<h3 class="section-header">Retinal Image</h3>', unsafe_allow_html=True)
            
            # Image upload
            uploaded_file = st.file_uploader("Upload a retinal fundus image", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Retinal Image", use_column_width=True)
                
                # Preprocess image
                preprocessed_image = preprocess_image(image)
                
                # Store in session state
                st.session_state.image = preprocessed_image
            else:
                # Use a sample image if available
                sample_image = st.checkbox("Use a sample image for demonstration")
                if sample_image:
                    # Load sample image (replace with your sample image path)
                    try:
                        image = Image.open("sample_retina.jpg")
                        st.image(image, caption="Sample Retinal Image", use_column_width=True)
                        
                        # Preprocess image
                        preprocessed_image = preprocess_image(image)
                        
                        # Store in session state
                        st.session_state.image = preprocessed_image
                    except:
                        st.warning("Sample image not found. Please upload your own image.")
        
        # Column 2: Clinical Data Input
        with col2:
            st.markdown('<h3 class="section-header">Clinical Data</h3>', unsafe_allow_html=True)
            
            # Create form for clinical data entry
            with st.form("clinical_data_form"):
                # Basic demographics
                age = st.number_input("Age", min_value=20, max_value=100, value=65)
                sex = st.radio("Gender", ["Female", "Male"])
                
                # Create two columns for clinical parameters
                col_a, col_b = st.columns(2)
                
                with col_a:
                    ejection_fraction = st.number_input("Ejection Fraction (%)", 10, 80, 45)
                    serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.5, 10.0, 1.2, step=0.1)
                    serum_sodium = st.number_input("Serum Sodium (mEq/L)", 110, 150, 137)
                    creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (CPK)", 10, 8000, 300)
                    platelets = st.number_input("Platelets (kiloplatelets/mL)", 100000, 850000, 250000, step=10000)
                
                with col_b:
                    diabetes = st.checkbox("Diabetes")
                    high_blood_pressure = st.checkbox("Hypertension")
                    anaemia = st.checkbox("Anemia")
                    smoking = st.checkbox("Smoking")
                    time = st.number_input("Follow-up time (days)", 4, 300, 60)
                
                # Submit button
                submit_button = st.form_submit_button("Analyze Risk")
                
                if submit_button:
                    # Create a dataframe with the clinical data
                    clinical_data = pd.DataFrame({
                        'age': [age],
                        'anaemia': [1 if anaemia else 0],
                        'creatinine_phosphokinase': [creatinine_phosphokinase],
                        'diabetes': [1 if diabetes else 0],
                        'ejection_fraction': [ejection_fraction],
                        'high_blood_pressure': [1 if high_blood_pressure else 0],
                        'platelets': [platelets],
                        'serum_creatinine': [serum_creatinine],
                        'serum_sodium': [serum_sodium],
                        'sex': [1 if sex == "Male" else 0],
                        'smoking': [1 if smoking else 0],
                        'time': [time]
                    })
                    
                    # Store in session state
                    st.session_state.clinical_data = clinical_data
                    
                    # Check if image is available
                    if hasattr(st.session_state, 'image'):
                        # Process the image and clinical data
                        with st.spinner("Analyzing... This may take a moment."):
                            # Get prediction
                            prediction = fusion_system.predict(
                                st.session_state.image, 
                                st.session_state.clinical_data
                            )
                            
                            # Get explanation
                            explanation = fusion_system.explain_prediction(
                                st.session_state.image, 
                                st.session_state.clinical_data,
                                prediction
                            )
                            
                            # Generate recommendations
                            recommendations = fusion_system.generate_recommendations(
                                explanation, 
                                st.session_state.clinical_data
                            )
                            
                            # Store results in session state
                            st.session_state.prediction = prediction
                            st.session_state.explanation = explanation
                            st.session_state.recommendations = recommendations
                            
                            # Switch to results tab
                            st.rerun()
                    else:
                        st.error("Please upload a retinal image or use the sample image.")
    
    # Tab 2: Results & Analysis
    with tab2:
        if hasattr(st.session_state, 'prediction') and hasattr(st.session_state, 'explanation'):
            st.markdown('<h2 class="sub-header">Cardiovascular Risk Assessment Results</h2>', unsafe_allow_html=True)
            
            # Get results from session state
            prediction = st.session_state.prediction
            explanation = st.session_state.explanation
            recommendations = st.session_state.recommendations
            
            # Display risk assessment
            risk_score = prediction['risk_score']
            risk_level = explanation['overall_risk_level']
            
            if risk_level == "High":
                risk_class = "risk-high"
            elif risk_level == "Moderate":
                risk_class = "risk-moderate"
            else:
                risk_class = "risk-low"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background-color: #f6f6f6; border-radius: 0.5rem; margin-bottom: 1rem;">
                <h3 style="color:black;">Overall Cardiovascular Risk: <span class="{risk_class}">{risk_level}</span></h3>
                <p style="font-size: 1.2rem;color:black;">Risk Score: <b>{risk_score:.2f}</b> (0-1 scale)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for results
            col1, col2, col3 = st.columns([1, 1, 1])
            
            # Column 1: Images
            with col1:
                st.markdown('<h3 class="section-header">Retinal Analysis</h3>', unsafe_allow_html=True)
                
                # Display original image
                st.image(st.session_state.image, caption="Original Retinal Image", use_column_width=True)
                
                # Display vessel segmentation
                if prediction['vessel_segmentation'] is not None:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.imshow(prediction['vessel_segmentation'], cmap=vessel_cmap)
                    ax.set_title("Vessel Segmentation")
                    ax.axis('off')
                    st.pyplot(fig)
            
            # Column 2: Risk Assessment
            with col2:
                st.markdown('<h3 class="section-header">Risk Breakdown</h3>', unsafe_allow_html=True)
                
                # Create risk gauge
                fig, ax = plt.subplots(figsize=(8, 5))
                risk_labels = ['Retinal Risk', 'Clinical Risk', 'Combined Risk']
                risk_values = [prediction['retinal_risk'], prediction['clinical_risk'], prediction['risk_score']]
                colors = ['#3498db', '#2ecc71', '#e74c3c']  # Blue, green, red
                
                # Create risk level bars
                bars = ax.barh(risk_labels, risk_values, color=colors)
                
                # Add value labels to the bars
                for i, (bar, value) in enumerate(zip(bars, risk_values)):
                    ax.text(min(value + 0.05, 0.95), i, f'{value:.2f}', va='center')
                
                # Add background zones for risk levels
                ax.axvspan(0, 0.33, alpha=0.1, color='green')
                ax.axvspan(0.33, 0.67, alpha=0.1, color='yellow')
                ax.axvspan(0.67, 1, alpha=0.1, color='red')
                
                # Add risk labels
                ax.text(0.15, -0.3, 'Low Risk', ha='center')
                ax.text(0.5, -0.3, 'Moderate Risk', ha='center')
                ax.text(0.85, -0.3, 'High Risk', ha='center')
                
                ax.set_xlim(0, 1)
                ax.set_title('Risk Assessment')
                ax.set_xlabel('Risk Score (0-1)')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Display key risk factors
                st.markdown('<h4>Key Risk Factors</h4>', unsafe_allow_html=True)
                
                if explanation['vessel_risk_factors']:
                    st.markdown('<h5>Retinal Risk Factors:</h5>', unsafe_allow_html=True)
                    for factor, value in explanation['vessel_risk_factors']:
                        st.write(f"• {factor}: {value:.2f}")
                else:
                    st.write("No significant retinal risk factors identified.")
                
                if explanation['clinical_risk_factors']:
                    st.markdown('<h5>Clinical Risk Factors:</h5>', unsafe_allow_html=True)
                    for factor, value in explanation['clinical_risk_factors']:
                        st.write(f"• {factor}: {value}")
                else:
                    st.write("No significant clinical risk factors identified.")
            
            # Column 3: Personalized Recommendations
            with col3:
                st.markdown('<h3 class="section-header">Personalized Recommendations</h3>', unsafe_allow_html=True)
                
                if recommendations:
                    st.markdown(recommendations)
                else:
                    st.write("Recommendations not available.")
        else:
            st.info("Please enter patient data and submit for analysis on the 'Patient Data Entry' tab.")
    
    # Tab 3: About the System
    with tab3:
        st.markdown('<h2 class="sub-header">About the Cardiovascular Risk Assessment System</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### How It Works
        
        This system uses a multi-modal approach to assess cardiovascular risk:
        
        1. **Retinal Analysis**: The retina is a unique window into cardiovascular health. Subtle changes in retinal
           blood vessels can reflect similar changes throughout the body, providing early warning signs of
           cardiovascular disease.
        
        2. **Clinical Assessment**: Traditional clinical measurements complement the retinal analysis, providing
           a more comprehensive picture of cardiovascular health.
        
        3. **Integrated Risk Score**: The system combines both assessments to provide a single, integrated risk score.
        
        4. **Personalized Recommendations**: An AI language model analyzes the results and provides personalized
           recommendations based on the identified risk factors.
        
        ### Key Components
        
        - **Vessel Segmentation Model**: A U-Net architecture that identifies and segments retinal blood vessels
        - **Retinal Risk Model**: An XGBoost classifier that predicts cardiovascular risk from vessel features
        - **Clinical Risk Model**: An XGBoost classifier that predicts heart failure risk from clinical parameters
        - **Fusion System**: A weighted ensemble that combines both risk assessments
        - **Gemini LLM**: A large language model that generates personalized recommendations
        
        ### Scientific Background
        
        Research has shown that changes in retinal vasculature can predict cardiovascular diseases with high accuracy.
        Key retinal biomarkers include:
        
        - **Vessel Density**: Lower vessel density can indicate decreased perfusion
        - **Vessel Tortuosity**: Increased twisting of vessels can indicate hypertension
        - **Vessel Caliber**: Narrower arterioles and wider venules indicate increased cardiovascular risk
        - **Vascular Patterns**: Abnormal branching patterns can indicate vascular dysfunction
        
        ### Disclaimer
        
        This system is for educational and research purposes only. It is not a substitute for professional
        medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider before
        making any changes to your healthcare regimen.
        """)
    
    # Footer
    st.markdown(
        """
        <div class="footer">
        Cardiovascular Health Assessment System &copy; 2025 | Built with Streamlit and TensorFlow
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()