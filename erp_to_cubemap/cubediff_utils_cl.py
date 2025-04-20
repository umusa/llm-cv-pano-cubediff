from cubediff_utils_v1 import *


def advanced_equirect_to_cubemap(equi, face_size):
    """
    Advanced high-precision equirectangular to cubemap conversion.
    
    Args:
        equi: Equirectangular panorama image
        face_size: Size of each cube face
        
    Returns:
        Dictionary of cubemap faces
    """
    import numpy as np
    import cv2
    
    H, W = equi.shape[:2]
    faces = {}
    
    # Use a higher internal processing resolution
    internal_size = int(face_size * 1.5)
    
    for name in ('front', 'right', 'back', 'left', 'top', 'bottom'):
        # Create grid for higher resolution
        y_coords, x_coords = np.meshgrid(
            (np.arange(internal_size) + 0.5) / internal_size * 2 - 1,
            (np.arange(internal_size) + 0.5) / internal_size * 2 - 1,
            indexing='ij'
        )
        
        # Calculate 3D ray direction
        if name == 'front':
            x, y, z = x_coords, -y_coords, np.ones_like(x_coords)
        elif name == 'right':
            x, y, z = np.ones_like(x_coords), -y_coords, -x_coords
        elif name == 'back':
            x, y, z = -x_coords, -y_coords, -np.ones_like(x_coords)
        elif name == 'left':
            x, y, z = -np.ones_like(x_coords), -y_coords, x_coords
        elif name == 'top':
            x, y, z = x_coords, np.ones_like(x_coords), y_coords
        elif name == 'bottom':
            x, y, z = x_coords, -np.ones_like(x_coords), -y_coords
        
        # Normalize vectors - simple and numerically stable approach
        norm = np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x/norm, y/norm, z/norm
        
        # Convert to spherical coordinates
        lon = np.arctan2(x, z)
        lat = np.arcsin(np.clip(y, -0.999999, 0.999999))
        
        # Convert to pixel coordinates with careful handling of edge cases
        map_x = ((lon / (2 * np.pi) + 0.5) * W - 0.5).astype(np.float32)
        map_y = ((0.5 - lat / np.pi) * H - 0.5).astype(np.float32)
        
        # Sample with appropriate interpolation
        face_img_high_res = cv2.remap(
            equi, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        
        # Downsample to requested face size
        face_img = cv2.resize(face_img_high_res, (face_size, face_size), 
                              interpolation=cv2.INTER_AREA)
        
        faces[name] = face_img
    
    return faces

def advanced_seam_handling(cube_faces):
    """
    Simple but effective seam handling that preserves image details.
    
    Args:
        cube_faces: Dictionary of cubemap faces
        
    Returns:
        Dictionary with improved seams
    """
    import numpy as np
    import cv2
    
    faces = {}
    for k, img in cube_faces.items():
        faces[k] = img.copy()
    
    face_size = next(iter(cube_faces.values())).shape[0]
    
    # Process adjacent face pairs
    pairs = [
        ('front', 'right'),
        ('right', 'back'),
        ('back', 'left'),
        ('left', 'front'),
        ('top', 'front'),
        ('top', 'right'),
        ('top', 'back'),
        ('top', 'left'),
        ('bottom', 'front'),
        ('bottom', 'right'),
        ('bottom', 'back'),
        ('bottom', 'left')
    ]
    
    # Use a modest blend width
    blend_width = face_size // 24
    
    # Simple linear blending for seams
    for face1, face2 in pairs:
        if face1 in ['top', 'bottom'] and face2 in ['front', 'right', 'back', 'left']:
            # Handle connections between top/bottom and side faces
            continue  # Skip for now as these are more complex
        
        # For side-to-side faces, use standard edge blending
        img1 = faces[face1].astype(np.float32)
        img2 = faces[face2].astype(np.float32)
        
        # Create linear weights for smooth transition
        weights = np.linspace(0, 1, blend_width)
        
        # Determine which edges to blend based on the face pair
        if (face1, face2) == ('front', 'right'):
            # Blend right edge of front with left edge of right
            for i in range(blend_width):
                alpha = weights[i]
                img1[:, -(i+1)] = (1 - alpha) * img1[:, -(i+1)] + alpha * img2[:, i]
                img2[:, i] = alpha * img1[:, -(i+1)] + (1 - alpha) * img2[:, i]
        
        elif (face1, face2) == ('right', 'back'):
            # Blend right edge of right with left edge of back
            for i in range(blend_width):
                alpha = weights[i]
                img1[:, -(i+1)] = (1 - alpha) * img1[:, -(i+1)] + alpha * img2[:, i]
                img2[:, i] = alpha * img1[:, -(i+1)] + (1 - alpha) * img2[:, i]
        
        elif (face1, face2) == ('back', 'left'):
            # Blend right edge of back with left edge of left
            for i in range(blend_width):
                alpha = weights[i]
                img1[:, -(i+1)] = (1 - alpha) * img1[:, -(i+1)] + alpha * img2[:, i]
                img2[:, i] = alpha * img1[:, -(i+1)] + (1 - alpha) * img2[:, i]
        
        elif (face1, face2) == ('left', 'front'):
            # Blend right edge of left with left edge of front
            for i in range(blend_width):
                alpha = weights[i]
                img1[:, -(i+1)] = (1 - alpha) * img1[:, -(i+1)] + alpha * img2[:, i]
                img2[:, i] = alpha * img1[:, -(i+1)] + (1 - alpha) * img2[:, i]
        
        # Update the faces
        faces[face1] = img1
        faces[face2] = img2
    
    return faces

def enhanced_back_face_processing(cube_faces):
    """
    Simple but effective processing focused on preserving details.
    
    Args:
        cube_faces: Dictionary of cubemap faces
        
    Returns:
        Dictionary with improved faces
    """
    import numpy as np
    import cv2
    
    faces = cube_faces.copy()
    
    # Minimal processing to avoid detail loss
    # Only focus on problematic faces: back, right, and poles
    for face_name in ['back', 'right']:
        face_img = faces[face_name].copy()
        
        # Subtle detail enhancement
        # Use unsharp mask to enhance details
        blurred = cv2.GaussianBlur(face_img, (0, 0), 1.5)
        sharpened = cv2.addWeighted(face_img, 1.5, blurred, -0.5, 0)
        
        # Apply very light bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(sharpened, d=3, sigmaColor=10, sigmaSpace=10)
        
        faces[face_name] = filtered
    
    # Special processing for pole faces
    for face_name in ['top', 'bottom']:
        face_img = faces[face_name].copy()
        
        # Apply a light, edge-preserving filter
        filtered = cv2.bilateralFilter(face_img, d=5, sigmaColor=20, sigmaSpace=20)
        
        # Enhance details
        blurred = cv2.GaussianBlur(filtered, (0, 0), 1.0)
        sharpened = cv2.addWeighted(filtered, 1.3, blurred, -0.3, 0)
        
        faces[face_name] = sharpened
    
    return faces

def advanced_cubemap_to_equirect(cube_faces, H, W):
    """
    Advanced cubemap to equirectangular conversion with precise sampling.
    
    Args:
        cube_faces: Dictionary of cubemap faces
        H, W: Height and width of output equirectangular image
        
    Returns:
        Equirectangular image
    """
    import numpy as np
    import cv2
    
    # Work at a slightly higher resolution for quality
    scale_factor = 1.2
    H_hires = int(H * scale_factor)
    W_hires = int(W * scale_factor)
    
    # Create latitude and longitude grid
    lat = np.linspace(np.pi/2, -np.pi/2, H_hires)
    lon = np.linspace(-np.pi, np.pi, W_hires)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Convert to Cartesian coordinates
    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)
    
    # Determine face for each pixel
    abs_x, abs_y, abs_z = np.abs(x), np.abs(y), np.abs(z)
    face_id = np.zeros_like(x, dtype=np.int32)
    
    # Front face (positive Z)
    face_id[(z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)] = 0
    
    # Right face (positive X)
    face_id[(x > 0) & (abs_x >= abs_z) & (abs_x >= abs_y)] = 1
    
    # Back face (negative Z)
    face_id[(z <= 0) & (abs_z >= abs_x) & (abs_z >= abs_y)] = 2
    
    # Left face (negative X)
    face_id[(x <= 0) & (abs_x >= abs_z) & (abs_x >= abs_y)] = 3
    
    # Top face (positive Y)
    face_id[(y > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)] = 4
    
    # Bottom face (negative Y)
    face_id[(y <= 0) & (abs_y >= abs_x) & (abs_y >= abs_z)] = 5
    
    # Compute local coordinates for each face
    face_size = next(iter(cube_faces.values())).shape[0]
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    
    # Small epsilon for numerical stability
    eps = 1e-9
    
    # Front face
    mask = face_id == 0
    u[mask] = x[mask] / (z[mask] + eps)
    v[mask] = -y[mask] / (z[mask] + eps)
    
    # Right face
    mask = face_id == 1
    u[mask] = -z[mask] / (x[mask] + eps)
    v[mask] = -y[mask] / (x[mask] + eps)
    
    # Back face
    mask = face_id == 2
    u[mask] = -x[mask] / (-z[mask] + eps)
    v[mask] = -y[mask] / (-z[mask] + eps)
    
    # Left face
    mask = face_id == 3
    u[mask] = z[mask] / (-x[mask] + eps)
    v[mask] = -y[mask] / (-x[mask] + eps)
    
    # Top face
    mask = face_id == 4
    u[mask] = x[mask] / (y[mask] + eps)
    v[mask] = z[mask] / (y[mask] + eps)
    
    # Bottom face
    mask = face_id == 5
    u[mask] = x[mask] / (-y[mask] + eps)
    v[mask] = -z[mask] / (-y[mask] + eps)
    
    # Clip values to ensure they're in range [-1,1]
    u = np.clip(u, -1, 1)
    v = np.clip(v, -1, 1)
    
    # Convert to pixel coordinates
    u_px = ((u + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    v_px = ((v + 1) * 0.5 * (face_size - 1)).astype(np.float32)
    
    # Sample from each face
    face_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
    equirect_hires = np.zeros((H_hires, W_hires, 3), dtype=np.uint8)
    
    # Create sampling maps for each face
    for face_idx, face_name in enumerate(face_names):
        mask = face_id == face_idx
        if not np.any(mask):
            continue
        
        map_x = np.zeros((H_hires, W_hires), dtype=np.float32)
        map_y = np.zeros((H_hires, W_hires), dtype=np.float32)
        
        map_x[mask] = u_px[mask]
        map_y[mask] = v_px[mask]
        
        face_img = cube_faces[face_name]
        sampled = cv2.remap(
            face_img, map_x, map_y,
            interpolation=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP
        )
        
        equirect_hires[mask] = sampled[mask]
    
    # Fix the ±π seam (additional post-processing)
    seam_width = W_hires // 50
    for i in range(seam_width):
        # Create a smooth blend weight
        weight = i / seam_width
        
        # Blend the left edge pixels with corresponding right edge pixels
        equirect_hires[:, i] = ((1-weight) * equirect_hires[:, i] + 
                              weight * equirect_hires[:, -(seam_width-i)]).astype(np.uint8)
        
        # Blend the right edge pixels with corresponding left edge pixels
        equirect_hires[:, -(i+1)] = ((1-weight) * equirect_hires[:, -(i+1)] + 
                                   weight * equirect_hires[:, seam_width-i-1]).astype(np.uint8)
    
    # Resize to final resolution
    equirect = cv2.resize(equirect_hires, (W, H), interpolation=cv2.INTER_AREA)
    
    return equirect


def optimized_equirect_cubemap_conversion(equirect_img, face_size=512):
    """
    Optimized end-to-end equirectangular to cubemap to equirectangular conversion
    pipeline focused on quality.
    
    Args:
        equirect_img: Input equirectangular image
        face_size: Size of cubemap faces (default: 512)
        
    Returns:
        tuple: (improved_equirect, cube_faces)
    """
    # Step 1: Convert to cubemap with high precision (minimal processing)
    cube_faces = advanced_equirect_to_cubemap(equirect_img, face_size)
    
    # Step 2: Apply minimal but effective back face processing
    # This step is done before seam handling to ensure the highest quality input
    cube_faces = enhanced_back_face_processing(cube_faces)
    
    # Step 3: Apply targeted seam handling
    cube_faces = advanced_seam_handling(cube_faces)
    
    # Step 4: Convert back to equirectangular with high precision
    improved_equirect = advanced_cubemap_to_equirect(cube_faces, equirect_img.shape[0], equirect_img.shape[1])
    
    return improved_equirect, cube_faces

def test_optimized_conversion(image_path, face_size=512):
    """
    Test and validate the optimized conversion pipeline.
    
    Args:
        image_path: Path to the equirectangular image
        face_size: Size of cubemap faces
        
    Returns:
        Dictionary containing metrics and visualization
    """
    import time
    
    # Load the image
    if image_path.startswith(('http://', 'https://')):
        equirect_img = load_image_from_url(image_path)
    else:
        equirect_img = load_local_image(image_path)
    
    if equirect_img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    print(f"Testing optimized conversion on image: {image_path}")
    print(f"Image shape: {equirect_img.shape}")
    
    # Run the benchmark conversion (for comparison)
    print("\nRunning benchmark method for comparison...")
    start_time = time.time()
    bench_equirect, bench_faces = high_quality_conversion(equirect_img, face_size)
    bench_time = time.time() - start_time
    print(f"Benchmark completed in {bench_time:.2f} seconds")
    
    # Calculate metrics for benchmark
    bench_metrics = diagnose_conversion_issues(equirect_img, bench_equirect)
    
    # Run our optimized conversion
    print("\nRunning optimized conversion method...")
    start_time = time.time()
    improved_equirect, improved_faces = optimized_equirect_cubemap_conversion(equirect_img, face_size)
    improved_time = time.time() - start_time
    print(f"Optimized method completed in {improved_time:.2f} seconds")
    
    # Calculate metrics for optimized method
    optimized_metrics = diagnose_conversion_issues(equirect_img, improved_equirect)
    
    # Print comparison
    print("\nBenchmark Method Metrics:")
    print(f"MSE: {bench_metrics['MSE']:.2f}")
    print(f"PSNR: {bench_metrics['PSNR']:.2f} dB")
    print(f"SSIM: {bench_metrics['SSIM']:.4f}")
    
    # Add regional SSIM reports for benchmark
    print("\nBenchmark Regional SSIM:")
    print(f"  North Pole: {bench_metrics['North_Pole_SSIM']:.4f}")
    print(f"  South Pole: {bench_metrics['South_Pole_SSIM']:.4f}")
    print(f"  Equator: {bench_metrics['Equator_SSIM']:.4f}")
    print(f"  Seam: {bench_metrics['Seam_SSIM']:.4f}")
    print("\nBenchmark Face SSIMs:")
    for face, ssim in bench_metrics['Longitude_SSIMs'].items():
        print(f"  {face}: {ssim:.4f}")
    
    print("\nOptimized Method Metrics:")
    print(f"MSE: {optimized_metrics['MSE']:.2f}")
    print(f"PSNR: {optimized_metrics['PSNR']:.2f} dB")
    print(f"SSIM: {optimized_metrics['SSIM']:.4f}")
    
    # Add regional SSIM reports for optimized
    print("\nOptimized Regional SSIM:")
    print(f"  North Pole: {optimized_metrics['North_Pole_SSIM']:.4f}")
    print(f"  South Pole: {optimized_metrics['South_Pole_SSIM']:.4f}")
    print(f"  Equator: {optimized_metrics['Equator_SSIM']:.4f}")
    print(f"  Seam: {optimized_metrics['Seam_SSIM']:.4f}")
    print("\nOptimized Face SSIMs:")
    for face, ssim in optimized_metrics['Longitude_SSIMs'].items():
        print(f"  {face}: {ssim:.4f}")
    
    # Calculate improvement percentages
    mse_improvement = (bench_metrics['MSE'] - optimized_metrics['MSE']) / bench_metrics['MSE'] * 100
    psnr_improvement = optimized_metrics['PSNR'] - bench_metrics['PSNR']
    ssim_improvement = (optimized_metrics['SSIM'] - bench_metrics['SSIM']) / bench_metrics['SSIM'] * 100
    
    # Calculate region-specific SSIM improvements
    north_ssim_improvement = (optimized_metrics['North_Pole_SSIM'] - bench_metrics['North_Pole_SSIM']) / bench_metrics['North_Pole_SSIM'] * 100
    south_ssim_improvement = (optimized_metrics['South_Pole_SSIM'] - bench_metrics['South_Pole_SSIM']) / bench_metrics['South_Pole_SSIM'] * 100
    equator_ssim_improvement = (optimized_metrics['Equator_SSIM'] - bench_metrics['Equator_SSIM']) / bench_metrics['Equator_SSIM'] * 100
    seam_ssim_improvement = (optimized_metrics['Seam_SSIM'] - bench_metrics['Seam_SSIM']) / bench_metrics['Seam_SSIM'] * 100
    
    # Find the most and least improved regions
    region_improvements = {
        'North Pole': north_ssim_improvement,
        'South Pole': south_ssim_improvement,
        'Equator': equator_ssim_improvement,
        'Seam': seam_ssim_improvement
    }
    
    most_improved_region = max(region_improvements.items(), key=lambda x: x[1])
    least_improved_region = min(region_improvements.items(), key=lambda x: x[1])
    
    print("\nImprovement Summary:")
    print(f"MSE Reduction: {mse_improvement:.1f}%")
    print(f"PSNR Improvement: {psnr_improvement:.2f} dB")
    print(f"SSIM Improvement: {ssim_improvement:.1f}%")
    print(f"Most Improved Region: {most_improved_region[0]} (+{most_improved_region[1]:.1f}%)")
    print(f"Least Improved Region: {least_improved_region[0]} (+{least_improved_region[1]:.1f}%)")
    
    # Get regions with issues from diagnoses
    print("\nDiagnosed Issues:")
    for diagnosis in optimized_metrics['Diagnoses']:
        print(f"- {diagnosis}")
    
    # Check if we met the targets - now including SSIM
    targets_met = all([
        optimized_metrics['MSE'] < 100,
        optimized_metrics['PSNR'] > 30,
        optimized_metrics['SSIM'] > 0.9,
        optimized_metrics['North_Pole_SSIM'] > 0.85,
        optimized_metrics['South_Pole_SSIM'] > 0.85,
        optimized_metrics['Seam_SSIM'] > 0.85,
        min(optimized_metrics['Longitude_SSIMs'].values()) > 0.87
    ])
    
    print(f"\nTargets Met: {targets_met}")
    if targets_met:
        print("✓ All quality targets have been achieved!")
    else:
        print("✗ Some quality targets have not been met:")
        if optimized_metrics['MSE'] >= 100:
            print(f"  - MSE target not met: {optimized_metrics['MSE']:.2f} >= 100")
        if optimized_metrics['PSNR'] <= 30:
            print(f"  - PSNR target not met: {optimized_metrics['PSNR']:.2f} <= 30 dB")
        if optimized_metrics['SSIM'] <= 0.9:
            print(f"  - SSIM target not met: {optimized_metrics['SSIM']:.4f} <= 0.9")
        if optimized_metrics['North_Pole_SSIM'] <= 0.85:
            print(f"  - North pole SSIM target not met: {optimized_metrics['North_Pole_SSIM']:.4f} <= 0.85")
        if optimized_metrics['South_Pole_SSIM'] <= 0.85:
            print(f"  - South pole SSIM target not met: {optimized_metrics['South_Pole_SSIM']:.4f} <= 0.85")
        if optimized_metrics['Seam_SSIM'] <= 0.85:
            print(f"  - Seam SSIM target not met: {optimized_metrics['Seam_SSIM']:.4f} <= 0.85")
        
        min_face_ssim = min(optimized_metrics['Longitude_SSIMs'].values())
        worst_face = min(optimized_metrics['Longitude_SSIMs'].items(), key=lambda x: x[1])[0]
        if min_face_ssim <= 0.87:
            print(f"  - Face SSIM target not met: {worst_face} face has SSIM of {min_face_ssim:.4f} <= 0.87")
    
    # Check speed improvement
    speed_improvement = (bench_time - improved_time) / bench_time * 100
    print(f"\nSpeed Improvement: {speed_improvement:.1f}%")
    
    # Calculate speedup ratio
    speedup = bench_time / improved_time
    print(f"Speedup Ratio: {speedup:.2f}x")
    
    # Visualize the results - FIXING THE ERROR HERE by removing the additional arguments
    visualize_conversion_comparison(
        equirect_img, 
        bench_equirect, 
        improved_equirect, 
        bench_faces, 
        improved_faces
    )
    
    # Display metrics info after visualization
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.title('SSIM Comparison by Region')
    regions = ['North_Pole', 'South_Pole', 'Equator', 'Seam']
    labels = ['North Pole', 'South Pole', 'Equator', 'Seam']
    benchmark_values = [bench_metrics[r + '_SSIM'] for r in regions]
    optimized_values = [optimized_metrics[r + '_SSIM'] for r in regions]
    
    x = range(len(regions))
    width = 0.35
    plt.bar([i - width/2 for i in x], benchmark_values, width, label='Benchmark')
    plt.bar([i + width/2 for i in x], optimized_values, width, label='Optimized')
    plt.ylabel('SSIM')
    plt.xticks(x, labels)
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(2, 1, 2)
    plt.title('SSIM Comparison by Cubemap Face')
    faces = list(bench_metrics['Longitude_SSIMs'].keys())
    benchmark_values = [bench_metrics['Longitude_SSIMs'][face] for face in faces]
    optimized_values = [optimized_metrics['Longitude_SSIMs'][face] for face in faces]
    
    x = range(len(faces))
    plt.bar([i - width/2 for i in x], benchmark_values, width, label='Benchmark')
    plt.bar([i + width/2 for i in x], optimized_values, width, label='Optimized')
    plt.ylabel('SSIM')
    plt.xticks(x, faces)
    plt.ylim(0.7, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
    # Return comprehensive report with all metrics
    return {
        'benchmark_metrics': bench_metrics,
        'optimized_metrics': optimized_metrics,
        'benchmark_faces': bench_faces,
        'optimized_faces': improved_faces,
        'benchmark_recon': bench_equirect,
        'optimized_recon': improved_equirect,
        'execution_time': {
            'benchmark': bench_time,
            'optimized': improved_time,
            'speedup': speedup,
            'improvement_percentage': speed_improvement
        },
        'improvement': {
            'mse': mse_improvement,
            'psnr': psnr_improvement,
            'ssim': ssim_improvement,
            'regional_ssim': region_improvements,
            'most_improved': most_improved_region[0],
            'least_improved': least_improved_region[0]
        },
        'targets_met': targets_met,
        'diagnoses': optimized_metrics['Diagnoses']
    }


def visualize_ssim_maps(original, benchmark, optimized):
    """
    Visualize SSIM maps to show where quality improvements are located.
    
    Args:
        original: Original equirectangular image
        benchmark: Benchmark reconstruction
        optimized: Optimized reconstruction
    """
    # Import necessary libraries
    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from matplotlib.colors import Normalize
    
    # Ensure images are in tensor format
    is_numpy = isinstance(original, np.ndarray)
    if is_numpy:
        if len(original.shape) == 3 and original.shape[2] <= 4:  # HWC format
            original_tensor = torch.from_numpy(original.transpose(2, 0, 1)).float()
            benchmark_tensor = torch.from_numpy(benchmark.transpose(2, 0, 1)).float()
            optimized_tensor = torch.from_numpy(optimized.transpose(2, 0, 1)).float()
        else:
            original_tensor = torch.from_numpy(original).float()
            benchmark_tensor = torch.from_numpy(benchmark).float()
            optimized_tensor = torch.from_numpy(optimized).float()
    else:
        original_tensor = original
        benchmark_tensor = benchmark
        optimized_tensor = optimized
    
    # Ensure images are in [0, 255] range
    if original_tensor.max() <= 1.0:
        original_tensor *= 255
        benchmark_tensor *= 255
        optimized_tensor *= 255
    
    # Define SSIM calculation function (simplified version for visualization)
    def calculate_ssim_map(img1, img2, window_size=11):
        # Constants for stabilization
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        
        # Generate a 1D Gaussian kernel
        sigma = 1.5
        gauss = torch.exp(torch.tensor([-(x - window_size//2)**2/(2*sigma**2) for x in range(window_size)]))
        gauss = gauss / gauss.sum()
        
        # Create 2D kernel by outer product
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        
        # Expand kernel for multi-channel images
        channels = img1.shape[0]
        window = _2D_window.expand(channels, 1, window_size, window_size).to(img1.device)
        
        # Pad images for convolution
        padding = window_size // 2
        
        # Function to calculate mean using convolution
        def conv_mean(x, window):
            return torch.nn.functional.conv2d(x, window, padding=padding, groups=channels)
            
        # Calculate means and variances
        mu1 = conv_mean(img1, window)
        mu2 = conv_mean(img2, window)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = conv_mean(img1 * img1, window) - mu1_sq
        sigma2_sq = conv_mean(img2 * img2, window) - mu2_sq
        sigma12 = conv_mean(img1 * img2, window) - mu1_mu2
        
        # SSIM calculation
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                  ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Return SSIM map (average over channels)
        return ssim_map.mean(dim=1, keepdim=True)
    
    # Calculate SSIM maps
    bench_ssim_map = calculate_ssim_map(original_tensor, benchmark_tensor)
    opt_ssim_map = calculate_ssim_map(original_tensor, optimized_tensor)
    
    # Calculate difference map
    improvement_map = opt_ssim_map - bench_ssim_map
    
    # Convert to numpy for visualization
    bench_ssim_np = bench_ssim_map.squeeze().cpu().numpy()
    opt_ssim_np = opt_ssim_map.squeeze().cpu().numpy()
    improvement_np = improvement_map.squeeze().cpu().numpy()
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot benchmark SSIM map
    im1 = axes[0].imshow(bench_ssim_np, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Benchmark SSIM Map')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot optimized SSIM map
    im2 = axes[1].imshow(opt_ssim_np, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Optimized SSIM Map')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot improvement map
    # Use a symmetric colormap centered at 0
    norm = Normalize(vmin=-0.2, vmax=0.2)
    im3 = axes[2].imshow(improvement_np, cmap='coolwarm', norm=norm)
    axes[2].set_title('SSIM Improvement Map (Blue: Worse, Red: Better)')
    plt.colorbar(im3, ax=axes[2])
    
    # Add labels
    for ax in axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xticks([0, original_tensor.shape[2]//4, original_tensor.shape[2]//2, 
                      3*original_tensor.shape[2]//4, original_tensor.shape[2]-1])
        ax.set_xticklabels(['0°', '90°', '180°', '270°', '360°'])
        ax.set_yticks([0, original_tensor.shape[1]//2, original_tensor.shape[1]-1])
        ax.set_yticklabels(['90°N', '0°', '90°S'])
    
    plt.tight_layout()
    plt.show()

