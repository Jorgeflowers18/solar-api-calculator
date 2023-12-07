import requests
import re
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io, color, measure
import rasterio
from rasterio.io import MemoryFile
import math
import streamlit as st
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from io import BytesIO
from PIL import Image

SOLAR_API_KEY = "AIzaSyCUu56Hv5fCnjV-ExShp3Z0JS-IuMYyqyI"
GEOLOCATION_API_KEY = 'AIzaSyCHMyD6qCmkgijFpbbRXHanxffZjJnFBxE'

building_insights_base_url = 'https://solar.googleapis.com/v1/buildingInsights:findClosest?'
data_layers_base_url = 'https://solar.googleapis.com/v1/dataLayers:get?'


def get_lat_lng(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={GEOLOCATION_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            latitude = data['results'][0]['geometry']['location']['lat']
            longitude = data['results'][0]['geometry']['location']['lng']
            return latitude, longitude
        else:
            print(f"Error: {data['status']}")
            print(response.status_code)
            return None, None
    else:
        print(f"Failed to get data: {response.status_code}")
        return None, None

def get_google_maps_image(lat, lon, zoom=19, size="600x600", maptype="satellite", api_key=GEOLOCATION_API_KEY):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": size,
        "maptype": maptype,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return image

get_lat_lng('Madrid, España')
def display_monthly_flux(data_layers, api_key):
    # For monthly flux, you need to handle multiple bands as it contains data for each month
    monthly_flux_url_with_key = f"{data_layers['monthlyFluxUrl']}&key={api_key}"
    response = requests.get(monthly_flux_url_with_key)
    if response.status_code == 200:
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                for i in range(1, dataset.count + 1):  # Loop through each band
                    fig, ax = plt.subplots()  # Create a new matplotlib figure and axes
                    band = dataset.read(i)
                    im = ax.imshow(band) #cmap='viridis')  # Store the mappable object in im
                    #plt.colorbar(im, ax=ax)  # Pass the mappable object to colorbar
                    ax.set_title(f"Monthly Solar Flux - Month {i}")
                    st.pyplot(fig)  # Pass the matplotlib figure to st.pyplot()
                    plt.close(fig)  # Close the figure
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")

# Function to display all bands of a GeoTIFF file from a URL with annotation
def display_all_geotiff_bands(url, api_key, title):
    url_with_key = f"{url}&key={api_key}"
    response = requests.get(url_with_key)
    if response.status_code == 200:
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                fig, ax = plt.subplots()  # Create a new matplotlib figure and axes
                if dataset.count > 1:
                    band = dataset.read([1, 2, 3])
                    band = np.transpose(band, (1, 2, 0))
                else:
                    band = dataset.read(1)
                    band = band.squeeze()

                im = ax.imshow(band)  # Store the mappable object in im, cmap='viridis' if dataset.count == 1 else None
                #plt.colorbar(im, ax=ax)  # Pass the mappable object to colorbar
                ax.set_title(title)
                st.pyplot(fig)  # Pass the matplotlib figure to st.pyplot()
                plt.close(fig)  # Close the figure

# Function to get data layers
@st.cache_data
def get_data_layers(lat, lon, radius=50, view="FULL_LAYERS", quality="LOW", pixel_size=0.5):
    url = f"https://solar.googleapis.com/v1/dataLayers:get"
    params = {
        "location.latitude": lat,
        "location.longitude": lon,
        "radiusMeters": radius,
        "view": view,
        "requiredQuality": quality,
        "pixelSizeMeters": pixel_size,
        "key": SOLAR_API_KEY  # Replace with your actual Google Solar API key
    }
    response = requests.get(url, params=params)
    return response.json()

# 2D visualization
def plot_roof_segments(roof_segments):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot each segment
    for segment in roof_segments:
        sw = segment['boundingBox']['sw']
        ne = segment['boundingBox']['ne']
        center = segment['center']

        # Calculate the width and height of the rectangle
        width = ne['longitude'] - sw['longitude']
        height = ne['latitude'] - sw['latitude']

        # Create a rectangle representing the roof segment
        rect = patches.Rectangle((sw['longitude'], sw['latitude']), width, height, 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        # Use azimuth to represent the orientation of the roof segment
        azimuth = segment['azimuthDegrees']
        pitch = segment['pitchDegrees']

        # Calculate the endpoint for the line indicating the orientation
        # Here, we're using a simple method to calculate the endpoint; you might need to adjust this
        end_lon = center['longitude'] + np.cos(np.radians(azimuth)) * width * 0.5
        end_lat = center['latitude'] + np.sin(np.radians(azimuth)) * height * 0.5

        # Draw a line indicating the orientation of the roof segment
        ax.plot([center['longitude'], end_lon], [center['latitude'], end_lat], 'k-')

        # Optionally, annotate the pitch of the roof segment
        ax.text(center['longitude'], center['latitude'], f"{pitch:.1f}°", 
                fontsize=9, ha='center', va='center')

    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Roof Segments with Orientation and Pitch')

    # Adjust plot limits
    ax.set_xlim(min([seg['boundingBox']['sw']['longitude'] for seg in roof_segments]),
                max([seg['boundingBox']['ne']['longitude'] for seg in roof_segments]))
    ax.set_ylim(min([seg['boundingBox']['sw']['latitude'] for seg in roof_segments]),
                max([seg['boundingBox']['ne']['latitude'] for seg in roof_segments]))

    # Show plot
    st.pyplot(fig)

# 3d visualization
def plot_3d_roof_segments(roof_segments):
    if not roof_segments:
        st.write("No roof segments data available.")
        return

    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    for segment in roof_segments:
        sw = segment['boundingBox']['sw']
        ne = segment['boundingBox']['ne']
        height = segment['planeHeightAtCenterMeters']
        pitch = math.radians(segment['pitchDegrees'])
        azimuth = math.radians(segment['azimuthDegrees'])

        # Debugging: Print pitch and azimuth
        #st.write(f"Pitch: {pitch} radians, Azimuth: {azimuth} radians")

        # Calculate the height difference based on pitch and azimuth
        delta_height = np.tan(pitch) * np.sqrt((ne['longitude'] - sw['longitude'])**2 + (ne['latitude'] - sw['latitude'])**2) 
        delta_x = delta_height * np.sin(azimuth)
        delta_y = delta_height * np.cos(azimuth)

        vertices = np.array([[sw['longitude'], sw['latitude'], height],
                             [ne['longitude'], sw['latitude'], height + delta_x],
                             [ne['longitude'], ne['latitude'], height + np.sqrt(delta_x**2 + delta_y**2)],
                             [sw['longitude'], ne['latitude'], height + delta_y]])

        # Debugging: Print vertices
        #st.write(vertices)

        # Create faces
        faces = [[vertices[j] for j in [0, 1, 2, 3]]]

        poly = Poly3DCollection(faces, alpha=0.5, edgecolors='k', linewidths=1)
        ax.add_collection3d(poly)

    # Set axes limits and labels
    ax.set_xlim(min(segment['boundingBox']['sw']['longitude'] for segment in roof_segments),
                max(segment['boundingBox']['ne']['longitude'] for segment in roof_segments))
    ax.set_ylim(min(segment['boundingBox']['sw']['latitude'] for segment in roof_segments),
                max(segment['boundingBox']['ne']['latitude'] for segment in roof_segments))
    ax.set_zlim(0, max(segment['planeHeightAtCenterMeters'] for segment in roof_segments) + 20)  # Adjusted z-limits

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Height')
    ax.set_title('3D Visualization of Roof Segments')

    # Show plot in Streamlit
    st.pyplot(fig)
def main():
    st.title('Solar Panel Insights')

    address = st.text_input("Enter your address:")

    # If data is in session_state, display it
    if 'data' in st.session_state:
        data = st.session_state.data
        lat, lng = get_lat_lng(st.session_state.address)

        # Display the image of the house
        image = get_google_maps_image(lat, lng)
        if 'imageryDate' in data:
            st.image(image, caption=f"House Image from {data['imageryDate']['year']}-{data['imageryDate']['month']}-{data['imageryDate']['day']}", use_column_width=True)
        else:
            st.image(image, caption="House Image", use_column_width=True)

        # Display solar data
        st.subheader('Solar Potential')
        # Add a note using st.markdown
        st.markdown("_Disclaimer: Based on Google Solar API data (panel height - 1.65m, panel width - 0.99m, 250 Watts)._")
        st.markdown(f"_Coordinates of the address: {lat} {lng}_")
        st.write(f"Max Array Panels Count: {data['solarPotential']['maxArrayPanelsCount']}")
        st.write(f"Max Solar Panel Area (m^2): {data['solarPotential']['maxArrayAreaMeters2']}")
        st.write(f"Total Roof Area (m^2): {data['solarPotential']['wholeRoofStats']['areaMeters2']}")
        st.write(f"Roof Segments: {len(data['solarPotential']['roofSegmentStats'])}")
        st.write(f"Max Sunshine Hours Per Year: {data['solarPotential']['maxSunshineHoursPerYear']}")

        if 'roofSegmentStats' in data['solarPotential']:
            st.subheader('Roof Segments 2D Visualization')
            plot_roof_segments(data['solarPotential']['roofSegmentStats'])

            st.subheader('Roof Segments 3D Visualization')
            roof_segments = data['solarPotential']['roofSegmentStats']                    
            plot_3d_roof_segments(roof_segments)

        # Calculator
        solar_calculator(data)

        # expander with full json
        with st.expander("Click to unfold the full solarAPI response", expanded=False):
            st.write(data)

        if st.button('Get Data Layers'):            
            # Fetch the data layers
            data_layers = get_data_layers(lat, lng)  # Make sure to implement this function to call the second API and get the data layers

            # Display the DSM, RGB, and Mask GeoTIFF files
            st.subheader('Digital Surface Model')
            display_all_geotiff_bands(data_layers['dsmUrl'], SOLAR_API_KEY, 'Digital Surface Model')

            st.subheader('RGB Composite Layer')
            display_all_geotiff_bands(data_layers['rgbUrl'], SOLAR_API_KEY, 'RGB Composite Layer')

            st.subheader('Building Mask')
            display_all_geotiff_bands(data_layers['maskUrl'], SOLAR_API_KEY, 'Building Mask')

            # Display the Annual Flux GeoTIFF file
            st.subheader('Annual Flux')
            display_all_geotiff_bands(data_layers['annualFluxUrl'], SOLAR_API_KEY, 'Annual Flux')

            # Display the Monthly Flux GeoTIFF files
            st.subheader('Monthly Flux')
            display_monthly_flux(data_layers, SOLAR_API_KEY)

            with st.expander("Click to unfold the full dataLayer response", expanded=False):
                st.write(data_layers)