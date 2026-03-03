import numpy as np
from scipy import ndimage, signal, sparse
import zarr
from tqdm import tqdm
import dask
import dask.array as da
import sys
import os
import scipy.io as sio
import tifffile as tif
from datetime import datetime
from tqdm import tqdm


def context_region(clnmsk, pix_pad=0):
    n,m = clnmsk.shape
    rows = np.any(clnmsk, axis=1)
    cols = np.any(clnmsk, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    top = rmin-pix_pad
    if top < 0:
        top = 0
    bot = rmax+pix_pad
    if bot > n:
        bot = n
    lef = cmin-pix_pad
    if lef < 0:
        lef = 0
    rig = cmax+pix_pad
    if rig > m:
        rig = m
    
    return top,bot,lef,rig

def pb_correct(im_data,fs):
    medn = np.median(im_data,axis=0)
    im_data = (im_data - medn[np.newaxis,:])
    bfilt,afilt = signal.butter(3, 1/3,'highpass', fs=fs)
    clean_data = signal.filtfilt(bfilt,afilt,im_data, axis=0)    
    return clean_data

def define_background(imageData, box):
    # Define Background
    n,m = imageData[1].shape
    height = 2*int((box[1] - box[0]))
    width = 2*int((box[3] - box[2]))
    t1 = box[0]-(height)
    t2 = box[0]-(height+1)
    b1 = box[1]+(height)
    b2 = box[1]+(height+1)
    l1 = box[2]-(width)
    l2 = box[2]-(width+1)
    r1 = box[3]+(width)
    r2 = box[3]+(width+1)

    if t1 < 0:
        t1 = 0
    if t2 < 0:
        t2 = 0
    if b1 > n:
        b1 = n
    if b2 > n:
        b2 = n
    if l1 < 0 :
        l1 = 0
    if l2 < 0:
        l2 = 0
    if r1 > m:
        r1 = m
    if r2 > m:
        r2 = m
        
    bg_top = imageData[:,t2:t1,l2:r2]
    bg_bot = imageData[:,b1:b2,l2:r2]
    bg_lef = imageData[:,t1:b1,l2:l1]
    bg_rig = imageData[:,t1:b1,r1:r2]
    
    bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_bot.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
    
    if b1 == b2:
        bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
    
    if t1 == t2:
        bg_pix = np.concatenate((bg_bot.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
        
    if l1 == l2:
        bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_bot.reshape(len(imageData),-1),bg_rig.reshape(len(imageData),-1)),axis=1)
    
    if r1 == r2:
        bg_pix = np.concatenate((bg_top.reshape(len(imageData),-1),bg_bot.reshape(len(imageData),-1),bg_lef.reshape(len(imageData),-1)),axis=1)
        
    #bg_pix = cp.array(bg_pix)
    
    return bg_pix

def background_svd(background_px, npc=8, lmd=0.01):
    n,m = background_px.shape
    #background_px = da.from_array(background_px)
    background_px = da.from_array(background_px,chunks=(n, 1))
    print(background_px)
    #U,Z,V = np.linalg.svd(background_px)
    #U, Z, V = da.linalg.svd_compressed(background_px,k=100)
    U, Z, V = da.linalg.svd(background_px)
    U.compute()
    U = np.array(U)
    Ub = U[:,0:npc]
    UbT = np.transpose(Ub)
    a = np.matmul(UbT,Ub)
    fnorm = np.sum(np.square(Ub))
    I = np.identity(npc)
    b = np.linalg.inv(a+lmd*fnorm*I)
    
    return Ub, UbT, b


def background_extraction(mask, context, bfilt, afilt, thresh):
    
    t,m,n = context.shape
    trace = context*mask[np.newaxis,:,:]
    trace = np.average(trace,axis=(1,2))
    trace = signal.filtfilt(bfilt,afilt,trace)
    th = np.std(np.abs(trace))*thresh
    spikeIdx = signal.find_peaks(trace,height=th)[0]
    spikeIdx = spikeIdx[(spikeIdx > 5) & (spikeIdx < t-5)]
    
    spk_mat = np.zeros((len(spikeIdx),m,n))
    
    for i in range(len(spikeIdx)):
        spk_mat[i,:,:] = context[spikeIdx[i]]-np.median(context[spikeIdx[i]-5:spikeIdx[i]+5,:,:],axis=0)
    
    background = np.average(spk_mat,axis=0)
    background = background/np.amax(background)
    background = np.where(background < 0.15,1,0)

    return background



def trace_extraction(fullData, ROIs, pole, fs, iters=6, thresh=6):

    # Bandpass Filter for Initial Spike Detection
    bfilt,afilt = signal.butter(3,13,'highpass',fs=fs)

    # Initialize Outputs
    final_masks = np.asarray([object]*len(ROIs), dtype=object)
    final_traces = np.asarray([object]*len(ROIs), dtype=object)
    final_spikes = np.asarray([object]*len(ROIs), dtype=object)
    final_spike_temps = np.asarray([object]*len(ROIs), dtype=object)
    final_snrs = np.zeros(len(ROIs))

    # Chunk in Time
    frame_chunk = 60*fs
    chunks = len(fullData)//frame_chunk
    if chunks < 1:
        chunks = 1

    for p in range(chunks):
        print("Processing Epoch "+str(p+1)+" of "+str(chunks))

        if p == chunks-1:
            imageData = fullData[p*frame_chunk:]
        else:
            imageData = fullData[p*frame_chunk:(p*frame_chunk)+frame_chunk]


        for i in range(len(ROIs)):
            print("Extracting Trace "+str(i+1)+" of "+str(len(ROIs)))
            t,n,m = imageData.shape

            # Focus on Region of Interest
            mask = ROIs[i]
            row_min, row_max, col_min, col_max = context_region(mask,pix_pad=2)
            c_pad = (col_max - col_min)
            r_pad = (row_max - row_min)

            if row_min-r_pad < 0:
                r_min_context = 0
            else:
                r_min_context = row_min - r_pad
            
            if row_max+r_pad > n:
                r_max_context = n
            else:
                r_max_context = row_max + r_pad

            if col_min-c_pad < 0:
                c_min_context = 0
            else:
                c_min_context = col_min - c_pad
            if col_max+c_pad > m:
                c_max_context = m
            else:
                c_max_context = col_max + c_pad

            mask = mask[r_min_context:r_max_context, c_min_context:c_max_context]
            context = imageData[:,r_min_context:r_max_context, c_min_context:c_max_context]
            tc,nc,mc = context.shape
            background = background_extraction(mask,context[0:2000,:,:],bfilt,afilt,thresh)
            
            if pole:
                context = -context

            context = context.reshape(len(context),-1)
            context = pb_correct(context,fs=fs)
            background = background.reshape(-1)
            background = np.where(background > 0)[0]

            if len(background) < 1000:
                background = np.copy(context[:,background])
            else:
                rnd = np.random.default_rng()
                rand_idx = rnd.integers(0, len(background), size=1000)
                background = np.copy(context[:,background[rand_idx]])

            Ub, UbT, b = background_svd(background)
            context = context.reshape(len(context),nc,mc)

            row_min, row_max, col_min, col_max = context_region(mask,pix_pad=2)
            context = context[:,row_min:row_max,col_min:col_max]
            mask = mask[row_min:row_max,col_min:col_max]

            print(context.shape)
            print(mask.shape)

            # Iterative Extraction
            for j in tqdm(range(iters+1)):

                # Extract first trace
                trace = context*mask[np.newaxis,:,:]
                trace = np.average(trace,axis=(1,2))
                
                # Background Subtraction
                beta = np.matmul(np.matmul(b,UbT),trace)
                trace = trace - np.matmul(Ub,beta)
                
                # Initial Spike Detection
                extract_spk_1 = np.copy(trace) - np.median(trace)
                extract_spk_1 = signal.filtfilt(bfilt,afilt,extract_spk_1)
                th = np.std(np.abs(extract_spk_1))*thresh
                spikeIdx = signal.find_peaks(extract_spk_1,height=th)[0]
                
                # # Create Initial Spike Template
                spk_win = 5
                spikes = spikeIdx[(spikeIdx > 10) & (spikeIdx < len(extract_spk_1)-10)]
                spk_mat = np.zeros((len(spikes),(2*spk_win)+1))
                
                for k in range(len(spikes)):
                    spk = np.copy(trace[spikes[k]-(spk_win-1):spikes[k]+spk_win+2])
                    spk_mat[k] = spk - np.median(spk)

                spk_template = np.average(spk_mat,axis=0)
                
                spiketrain = np.zeros(len(trace))
                spiketrain[spikeIdx] = 1
                qq = signal.convolve(spiketrain, np.ones((2*spk_win)+1),'same')
                noise = trace[qq<0.5]

                # Signal pre-whiten from welch spectral analysis
                freq,pxx = signal.welch(noise,nfft=len(trace))
                pxx = np.sqrt(pxx)
                
                if len(trace)%2 == 0:
                    pxx = np.append(pxx[:-1],np.flip(pxx[:-1]))
                else:
                    pxx = np.append(pxx[:-1],np.flip(pxx)) 
                
                # Pre-whiten signal
                extract_spk_1 = np.fft.fft(np.copy(trace))/pxx
                extract_spk_1 = np.real(np.fft.ifft(extract_spk_1))

                spk_mat = np.zeros((len(spikes),(2*spk_win)+1))

                for k in range(len(spikes)):
                    spk = np.copy(extract_spk_1[spikes[k]-(spk_win-1):spikes[k]+spk_win+2])
                    spk_mat[k] = spk - np.median(spk)
                    
                # Match filter spike extraction from whitened signal
                match_temp = np.average(spk_mat,axis=0)
                extract_spk_1 = signal.correlate(extract_spk_1,match_temp,'same') 
                th = np.std(np.abs(extract_spk_1))*(thresh*(0.75))
                spikeIdx = signal.find_peaks(extract_spk_1,height=th)[0]
                
                if len(spikeIdx) < 1:
                    snr = 0
                    print("No Spikes Detected")
                    break
                
                if j == range(iters+1)[-1]:    
                    spiketrain = np.zeros(len(trace))
                    spiketrain[spikeIdx] = 1
                    qq = signal.convolve(spiketrain, np.ones((2*spk_win+1)),'same')
                    sign = np.amax(spk_template) - np.amin(spk_template)
                    nois = np.std(trace[qq<0.5])
                    snr = sign/nois
                    break
                
                # Reconstruct trace from spike template
                spiketrain = np.zeros(len(trace))
                spiketrain[spikeIdx] = 1
                trec = signal.convolve(spiketrain,spk_template, 'same')

                # Refine Spatial Footprint
                context = context.reshape(len(context),-1)
                solver_params = sparse.linalg.lsmr(context,trec, damp=0.01, maxiter=1)
                mask = solver_params[0]
                mask = mask/np.amax(mask)
                mask[mask < 0.1] = 0
                mask = np.reshape(mask,(row_max-row_min,col_max-col_min))
                context = np.reshape(context,(len(context),row_max-row_min,col_max-col_min))

            r_min, r_max, c_min, c_max = context_region(ROIs[i],pix_pad=2)
            canvas = np.zeros((imageData.shape[1],imageData.shape[2]))
            canvas[r_min:r_max, c_min:c_max] = mask

            final_masks[i] = canvas
            final_traces[i] = trace
            final_spikes[i] = spikeIdx
            final_spike_temps[i] = spk_template
            final_snrs[i] = snr
        

        if p == 0:
            all_masks = np.array([final_masks])
            all_traces = np.array([final_traces])
            all_spikes = np.array([final_spikes])
            all_spike_temps = np.array([final_spike_temps])
            all_snrs = np.array([final_snrs])
        else:
            all_masks = np.concatenate((all_masks, [final_masks]),axis=0)
            all_traces = np.concatenate((all_traces, [final_traces]),axis=0)
            all_spikes = np.concatenate((all_spikes, [final_spikes]),axis=0)
            all_spike_temps = np.concatenate((all_spike_temps, [final_spike_temps]),axis=0)
            all_snrs = np.concatenate((all_snrs, [final_snrs]),axis=0)

    results = {
        'Masks': all_masks,
        'DFF': all_traces,
        'Spikes': all_spikes,
        'SpikeTemplate': all_spike_temps,
        'SpikeSNR': all_snrs
        }
            
    return results
            
            
            
if __name__ == '__main__':
    reg_data_path = str(sys.argv[1])
    roi_data_path = str(sys.argv[2])
    pole = bool(int(sys.argv[3]))
    fs = int(sys.argv[4])
    output_data_path = str(sys.argv[5])

    registered_data = zarr.open(reg_data_path)
    rois = tif.imread(roi_data_path)

    results = trace_extraction(registered_data, rois, pole, fs)

    now = datetime.now()
    saved_dir = os.path.join(output_data_path, 'saved')
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    date_time = now.strftime("%Y-%m-%d_%H%M%S")
    output_file_name = "VIP_Saved_" + date_time
    np.save(os.path.join(saved_dir, output_file_name+'.npy'), results)
    sio.savemat(os.path.join(saved_dir, output_file_name+'.mat'), results)



        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    

    
