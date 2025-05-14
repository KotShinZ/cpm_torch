import torch
import numpy as np

# --- PyTorch版の関数 ---

def fade_pt(t: torch.Tensor) -> torch.Tensor:
    """5次多項式によるフェード関数 (PyTorch版)"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def lerp_pt(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """線形補間 (実際にはfade関数を通したtによる滑らかな補間) (PyTorch版)"""
    return a + fade_pt(t) * (b - a)

def perlin_pt(r_coords: torch.Tensor, seed: int = None, device: torch.device = None) -> torch.Tensor:
    """
    Perlinノイズ生成関数 (PyTorch版)

    Args:
        r_coords (torch.Tensor): 座標を指定するテンソル。形状は (2, H, W) を期待。
                                 r_coords[0] は X座標、r_coords[1] は Y座標。
        seed (int, optional): 乱数シード。Defaults to None.
        device (torch.device, optional): 計算に使用するデバイス ('cpu' or 'cuda')。
                                         Noneの場合、r_coordsのデバイスを使用。
                                         Defaults to None.

    Returns:
        torch.Tensor: 生成されたPerlinノイズ。形状は (H, W)。
    """
    if device is None:
        device = r_coords.device
    
    r_coords = r_coords.to(device)

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == 'cuda': # GPU reproducibility
            torch.cuda.manual_seed_all(seed)


    # 整数部 (インデックスとして使用)
    ri = torch.floor(r_coords).long()
    
    # インデックスとして使用するための準備 (座標の最小値を0にする)
    # これにより、座標の絶対値に関わらず、相対的な位置で勾配グリッドを参照できる
    ri_x_min = ri[0].min()
    ri_y_min = ri[1].min()
    ri0_normalized = ri[0] - ri_x_min
    ri1_normalized = ri[1] - ri_y_min
    
    # 小数部
    rf = r_coords - torch.floor(r_coords) # または r_coords % 1.0

    # 格子点の勾配ベクトルをランダムに生成
    # グリッドサイズは入力座標の範囲に合わせる
    # インデックスとして riN_normalized.max() + 1 までアクセスするため、サイズは +2 する
    max_ri0 = ri0_normalized.max()
    max_ri1 = ri1_normalized.max()
    
    g_shape = (max_ri0.item() + 2, max_ri1.item() + 2, 2)
    g = (2 * torch.rand(g_shape, dtype=r_coords.dtype, device=device) - 1)

    # 四隅の相対座標 (0,0), (0,1), (1,0), (1,1)
    # e の形状: (1, 1, 4, 2)
    e = torch.tensor([[[[0,0],[0,1],[1,0],[1,1]]]], dtype=r_coords.dtype, device=device)

    # 各格子点から見た位置ベクトル (distance vectors)
    # rf の形状: (2, H, W) -> rf_permuted の形状: (H, W, 1, 2)
    # (rf_permuted - e) のブロードキャストにより diff_pt の形状: (H, W, 4, 2)
    # er の最終形状: (H, W, 4, 1, 2)
    H, W = r_coords.shape[1], r_coords.shape[2]
    rf_permuted = rf.permute(1,2,0).unsqueeze(2) # (H,W,2) -> (H,W,1,2)
    diff_pt = rf_permuted - e                   # (H,W,1,2) - (1,1,4,2) -> (H,W,4,2) via broadcasting
    er = diff_pt.reshape(H, W, 4, 1, 2)

    # 四隅の勾配ベクトルを取得
    # g00, g01, g10, g11 の形状: (H, W, 2)
    g00 = g[ri0_normalized, ri1_normalized]
    g01 = g[ri0_normalized, ri1_normalized + 1] # (i, j+1)
    g10 = g[ri0_normalized + 1, ri1_normalized] # (i+1, j)
    g11 = g[ri0_normalized + 1, ri1_normalized + 1] # (i+1, j+1)

    # これらをスタックして (H,W,4,2) の形状にし、内積計算のために次元追加して (H,W,4,2,1)
    # gr[h,w,0,:,0] = g00[h,w,:]
    # gr[h,w,1,:,0] = g01[h,w,:] (e[1]=[0,1]に対応)
    # gr[h,w,2,:,0] = g10[h,w,:] (e[2]=[1,0]に対応)
    # gr[h,w,3,:,0] = g11[h,w,:] (e[3]=[1,1]に対応)
    gr_stacked = torch.stack([g00, g01, g10, g11], dim=2) # (H, W, 4, 2)
    gr = gr_stacked.unsqueeze(-1)                         # (H, W, 4, 2, 1)

    # 勾配ベクトルと位置ベクトルの内積 (ドット積)
    # er: (H,W,4,1,2), gr: (H,W,4,2,1) -> p_matmul: (H,W,4,1,1)
    p_matmul = torch.matmul(er, gr)
    p_reshaped = p_matmul.squeeze(-1).squeeze(-1) # (H,W,4)
    
    # 補間のために (4,H,W) の形状に変換
    # p[0] = dot(rf-(0,0), g00), p[1] = dot(rf-(0,1), g01), ...
    p_dots = p_reshaped.permute(2,0,1) # (4,H,W)

    # rf[0] は x方向の小数部 (fx), rf[1] は y方向の小数部 (fy)
    fx = rf[0] # (H,W)
    fy = rf[1] # (H,W)

    # 線形補間 (実際にはfade関数による滑らかな補間)
    # p_dots[0] (top-left influence), p_dots[1] (bottom-left), p_dots[2] (top-right), p_dots[3] (bottom-right)
    # The original lerp sequence implies p[0], p[2] are x-neighbors, and p[1], p[3] are x-neighbors
    # lerp(p[0],p[2],rf[0]) means p[0] is influence at (i,j) and p[2] is influence at (i+1,j)
    # lerp(p[1],p[3],rf[0]) means p[1] is influence at (i,j+1) and p[3] is influence at (i+1,j+1)
    # This matches our p_dots construction:
    # p_dots[0] uses g00 (i,j), er uses (fx,fy)
    # p_dots[1] uses g01 (i,j+1), er uses (fx,fy-1)
    # p_dots[2] uses g10 (i+1,j), er uses (fx-1,fy)
    # p_dots[3] uses g11 (i+1,j+1), er uses (fx-1,fy-1)
    
    val0 = lerp_pt(p_dots[0], p_dots[2], fx) # Interpolate along x for y=0 cells
    val1 = lerp_pt(p_dots[1], p_dots[3], fx) # Interpolate along x for y=1 cells
    
    return lerp_pt(val0, val1, fy) # Interpolate along y

# --- PyTorch版 フラクタルPerlinノイズ (fBm) ---
def fractal_perlin_noise_pt(
    shape: tuple[int, int],
    scale_factor: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int = None,
    device: torch.device = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Generates 2D Fractal Brownian Motion (fBm) noise using PyTorch.

    Args:
        shape (tuple[int, int]): The (Height, Width) dimensions of the noise image.
        scale_factor (float): Base scale of the noise. Larger values result in
                              larger, lower-frequency patterns (zoomed-in appearance).
        octaves (int): Number of noise layers (octaves) to combine.
        persistence (float): Amplitude multiplier for each successive octave (0 < persistence < 1).
                             Controls how much successive octaves contribute to the final noise.
        lacunarity (float): Frequency multiplier for each successive octave (lacunarity > 1).
                            Controls how much detail (higher frequency) is added in successive octaves.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        device (torch.device, optional): PyTorch device ('cpu' or 'cuda'). 
                                         Defaults to 'cpu' if None.
        normalize (bool, optional): If True, normalizes the output to the range [-1, 1].
                                    Defaults to True.

    Returns:
        torch.Tensor: A 2D tensor of shape (H, W) containing the generated noise.
    """
    H, W = shape
    if device is None:
        # 入力shapeからテンソルを作らないので、明示的にCPUをデフォルトにするか、引数で指定必須にする
        device = torch.device('cpu') 

    total_noise = torch.zeros(H, W, dtype=torch.float32, device=device)
    current_amplitude = 1.0
    current_frequency_multiplier = 1.0  # Multiplies base coordinates to change frequency

    # Base coordinates (0..W-1 for x, 0..H-1 for y)
    # These are pixel indices, which will be scaled.
    x_indices = torch.arange(W, device=device, dtype=torch.float32)
    y_indices = torch.arange(H, device=device, dtype=torch.float32)
    
    # Create meshgrid: y_coords_mesh and x_coords_mesh will have shape (H, W)
    # y_coords_mesh[i,j] = y_indices[i]
    # x_coords_mesh[i,j] = x_indices[j]
    y_coords_mesh, x_coords_mesh = torch.meshgrid(y_indices, x_indices, indexing='ij')

    for i in range(octaves):
        # Scale coordinates for the current octave:
        # - Dividing by 'scale_factor': larger scale_factor -> coordinates passed to perlin_pt are smaller
        #   -> perlin noise samples from a smaller region of its "unit" grid -> appears zoomed in (larger features).
        # - Multiplying by 'current_frequency_multiplier': larger multiplier -> coordinates are larger
        #   -> perlin noise samples from a wider region -> appears zoomed out (smaller, more frequent features).
        scaled_x_coords = x_coords_mesh * current_frequency_multiplier / scale_factor
        scaled_y_coords = y_coords_mesh * current_frequency_multiplier / scale_factor
        
        # perlin_pt expects coordinates in shape (2, H, W), with X first, then Y.
        r_coords_octave = torch.stack([scaled_x_coords, scaled_y_coords])
        
        # Use a different seed for each octave to ensure varied patterns
        octave_seed = seed + i if seed is not None else None
        
        noise_octave = perlin_pt(r_coords_octave, seed=octave_seed, device=device)
        
        total_noise += noise_octave * current_amplitude
        
        # Update amplitude and frequency for the next octave
        current_amplitude *= persistence
        current_frequency_multiplier *= lacunarity
        
    if normalize:
        # Normalize the total noise to the range [-1, 1]
        min_val = total_noise.min()
        max_val = total_noise.max()
        if max_val > min_val: # Avoid division by zero if the noise is flat
            total_noise = 2.0 * (total_noise - min_val) / (max_val - min_val) - 1.0
        else:
            # If all values are the same (flat noise), map to 0.
            # (total_noise - min_val) would be all zeros.
            total_noise = torch.zeros_like(total_noise) 
            
    return total_noise

# --- 元のNumPy版の関数 (比較用) ---
def fade_np(t):return 6*t**5-15*t**4+10*t**3
def lerp_np(a,b,t):return a+fade_np(t)*(b-a)

def perlin_np(r,seed=np.random.randint(0,100)):
    np.random.seed(seed)

    ri = np.floor(r).astype(int)
    ri[0] -= ri[0].min()
    ri[1] -= ri[1].min()
    rf = np.array(r) % 1
    g = 2 * np.random.rand(ri[0].max()+2,ri[1].max()+2,2) - 1
    e = np.array([[[[0,0],[0,1],[1,0],[1,1]]]])
    er = (np.array([rf]).transpose(2,3,0,1) - e).reshape(r.shape[1],r.shape[2],4,1,2)
    g_indices_0 = ri[0]
    g_indices_1 = ri[1]
    g00 = g[g_indices_0,   g_indices_1]
    g01 = g[g_indices_0,   g_indices_1+1]
    g10 = g[g_indices_0+1, g_indices_1]
    g11 = g[g_indices_0+1, g_indices_1+1]

    # NumPyのnp.r_の複雑な部分を明示的なスタッキングに置き換えて意図を明確化
    # 元のコード: gr = np.r_["3,4,0",g00,g01,g10,g11].transpose(0,1,3,2).reshape(r.shape[1],r.shape[2],4,2,1)
    # この部分は、各ピクセルに対応する4つの勾配ベクトル (g00, g01, g10, g11) を集め、
    # (H,W,4,2,1) の形状に整形するのが目的。
    # g00,g01,g10,g11はそれぞれ(H,W,2)の形状。
    gr_stacked_np = np.stack([g00,g01,g10,g11], axis=2) # (H,W,4,2)
    gr = gr_stacked_np[..., np.newaxis] # (H,W,4,2,1)
    
    p = (er@gr).reshape(r.shape[1],r.shape[2],4).transpose(2,0,1)
    
    return lerp_np(lerp_np(p[0],p[2],rf[0]),lerp_np(p[1],p[3],rf[0]),rf[1])

