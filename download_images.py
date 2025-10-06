#!/usr/bin/env python3
"""
Script to download external images from markdown file and update paths to local references.
"""

import os
import re
import requests
import urllib.parse
from pathlib import Path
import time

def create_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def download_image(url, local_path):
    """Download image from URL to local path."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"✓ Downloaded: {url}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        return False

def extract_image_urls(markdown_content):
    """Extract all image URLs from markdown content."""
    # Pattern to match markdown images: ![](url) or ![alt](url)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    matches = re.findall(pattern, markdown_content)
    
    # Filter for external URLs (not starting with ../ or /)
    external_urls = []
    for alt_text, url in matches:
        if not (url.startswith('../') or url.startswith('/') or url.startswith('./')):
            external_urls.append((alt_text, url))
    
    return external_urls

def get_filename_from_url(url):
    """Extract filename from URL."""
    parsed_url = urllib.parse.urlparse(url)
    filename = os.path.basename(parsed_url.path)
    
    # If no extension, try to get it from the URL or default to .png
    if not os.path.splitext(filename)[1]:
        if 'png' in url.lower():
            filename += '.png'
        elif 'jpg' in url.lower() or 'jpeg' in url.lower():
            filename += '.jpg'
        elif 'gif' in url.lower():
            filename += '.gif'
        else:
            filename += '.png'  # default
    
    return filename

def update_markdown_content(content, url_mapping):
    """Update markdown content with new local image paths."""
    updated_content = content
    
    for old_url, new_path in url_mapping.items():
        # Replace both ![](url) and ![alt](url) patterns
        pattern1 = f'!\\[([^\\]]*)\\]\\({re.escape(old_url)}\\)'
        replacement1 = f'![\\1]({new_path})'
        updated_content = re.sub(pattern1, replacement1, updated_content)
    
    return updated_content

def main():
    # Configuration
    markdown_file = "content/posts/wm_2025.md"
    images_dir = "content/posts/img/wm_2025"
    
    print("🖼️  Image Downloader for World Models 2025 Post")
    print("=" * 50)
    
    # Create images directory
    create_directory(images_dir)
    print(f"📁 Created directory: {images_dir}")
    
    # Read markdown file
    try:
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"📖 Read markdown file: {markdown_file}")
    except Exception as e:
        print(f"✗ Error reading markdown file: {e}")
        return
    
    # Extract image URLs
    image_urls = extract_image_urls(content)
    print(f"🔍 Found {len(image_urls)} external image URLs")
    
    if not image_urls:
        print("ℹ️  No external images found to download.")
        return
    
    # Download images and create mapping
    url_mapping = {}
    successful_downloads = 0
    
    for i, (alt_text, url) in enumerate(image_urls, 1):
        print(f"\n[{i}/{len(image_urls)}] Processing: {url}")
        
        # Generate local filename
        filename = get_filename_from_url(url)
        local_path = os.path.join(images_dir, filename)
        
        # Handle duplicate filenames
        counter = 1
        original_path = local_path
        while os.path.exists(local_path):
            name, ext = os.path.splitext(original_path)
            local_path = f"{name}_{counter}{ext}"
            counter += 1
        
        # Download image
        if download_image(url, local_path):
            # Create relative path for markdown
            relative_path = f"../img/wm_2025/{os.path.basename(local_path)}"
            url_mapping[url] = relative_path
            successful_downloads += 1
        
        # Add small delay to be respectful to servers
        time.sleep(0.5)
    
    print(f"\n📊 Download Summary:")
    print(f"   Total URLs found: {len(image_urls)}")
    print(f"   Successfully downloaded: {successful_downloads}")
    print(f"   Failed downloads: {len(image_urls) - successful_downloads}")
    
    if successful_downloads > 0:
        # Update markdown file
        print(f"\n🔄 Updating markdown file...")
        updated_content = update_markdown_content(content, url_mapping)
        
        # Create backup
        backup_file = f"{markdown_file}.backup"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"💾 Created backup: {backup_file}")
        
        # Write updated content
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        print(f"✅ Updated markdown file: {markdown_file}")
        
        print(f"\n🎉 Successfully processed {successful_downloads} images!")
        print(f"📁 Images saved to: {images_dir}")
        print(f"📝 Markdown file updated with local paths")
    else:
        print("\n❌ No images were successfully downloaded.")

if __name__ == "__main__":
    main()
