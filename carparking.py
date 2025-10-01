import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Tuple, Dict

class ParkingDetectorFromJSON:
    """
    Parking detection using pre-defined slots from JSON
    Works directly with your parking_slots.json file
    """
    
    def __init__(self, video_path: str, output_dir: str = "output"):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.slots = []
        self.slot_status_all = []
    
    # ============ LOAD SLOTS FROM JSON ============
    
    def load_slots_from_json(self, json_path: str):
        """
        Load parking slots from JSON file
        Format: [[x, y, width, height], ...]
        """
        print(f"\nLoading parking slots from: {json_path}")
        
        with open(json_path, 'r') as f:
            self.slots = json.load(f)
        
        self.slot_status_all = [None] * len(self.slots)
        
        print(f"✓ Loaded {len(self.slots)} parking slots")
        
        return self.slots
    
    def visualize_loaded_slots(self, reference_frame_path: str = None):
        """
        Visualize loaded slots to verify they're correct
        """
        print("\n" + "=" * 60)
        print("VISUALIZING LOADED SLOTS")
        print("=" * 60)
        
        # Get reference frame
        if reference_frame_path is None:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Cannot read video")
        else:
            frame = cv2.imread(reference_frame_path)
        
        vis_frame = frame.copy()
        
        # Draw all slots
        for i, (x, y, w, h) in enumerate(self.slots):
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add slot number every 10 slots to avoid clutter
            if i % 10 == 0:
                cv2.putText(vis_frame, str(i+1), (x+5, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add info
        cv2.putText(vis_frame, f"Total Slots: {len(self.slots)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(vis_frame, f"Total Slots: {len(self.slots)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        # Save visualization
        vis_path = self.output_dir / "loaded_slots_preview.jpg"
        cv2.imwrite(str(vis_path), vis_frame)
        print(f"✓ Preview saved to: {vis_path}")
        
        # Display in resizable window
        cv2.namedWindow('Loaded Slots', cv2.WINDOW_NORMAL)
        
        # Try to get screen size
        try:
            import tkinter as tk
            root = tk.Tk()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
            
            window_w = int(screen_w * 0.9)
            window_h = int(screen_h * 0.9)
            cv2.resizeWindow('Loaded Slots', window_w, window_h)
        except:
            pass
        
        cv2.imshow('Loaded Slots', vis_frame)
        print("\nPress any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # ============ OCCUPANCY DETECTION ============
    
    def calc_diff(self, im1: np.ndarray, im2: np.ndarray) -> float:
        """Calculate difference between two images"""
        return np.abs(np.mean(im1) - np.mean(im2))
    
    def slot_availability_simple(self, slot_crop: np.ndarray, 
                                 edge_threshold: float = 0.08,
                                 variance_threshold: float = 2500) -> bool:
        """
        Improved simple detection using edge density and variance
        Works for all car colors including dark vehicles
        
        Args:
            slot_crop: Image crop of parking slot
            edge_threshold: Edge density threshold (occupied if > threshold)
            variance_threshold: Variance threshold (occupied if > threshold)
        
        Returns:
            True if slot is empty, False if occupied
        """
        if slot_crop.size == 0:
            return True
        
        if len(slot_crop.shape) == 3:
            gray = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = slot_crop
        
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Feature 1: Edge detection (cars have distinct edges)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Feature 2: Variance (texture variation)
        variance = np.var(gray)
        
        # Feature 3: Standard deviation for additional texture info
        std_dev = np.std(gray)
        
        # Combined scoring for better accuracy
        # A slot is occupied if it has EITHER:
        # 1. High edge density (car shape/boundaries) OR
        # 2. High variance AND moderate std_dev (textured surface)
        
        has_car_edges = edge_density > edge_threshold
        has_car_texture = variance > variance_threshold and std_dev > 25
        
        is_occupied = has_car_edges or has_car_texture
        is_empty = not is_occupied
        
        return is_empty
    
    def calibrate_threshold(self, reference_frame_path: str = None):
        """
        Analyze slot features to find optimal thresholds
        Helps tune edge_threshold and variance_threshold
        """
        print("\n" + "=" * 60)
        print("SLOT FEATURE ANALYSIS - SIMPLE METHOD")
        print("=" * 60)
        
        if not self.slots:
            print("No slots loaded. Load slots first.")
            return
        
        # Get reference frame
        if reference_frame_path is None:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Cannot read video")
        else:
            frame = cv2.imread(reference_frame_path)
        
        print("\nAnalyzing slot features...")
        print("\n{:<6} {:<12} {:<12} {:<12} {:<10}".format(
            'Slot', 'EdgeDensity', 'Variance', 'StdDev', 'Status'))
        print("-" * 60)
        
        # Analyze all slots to get statistics
        all_edge_densities = []
        all_variances = []
        all_std_devs = []
        
        for i, (x, y, w, h) in enumerate(self.slots):
            slot_crop = frame[y:y+h, x:x+w, :]
            
            if len(slot_crop.shape) == 3:
                gray = cv2.cvtColor(slot_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = slot_crop
            
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Calculate features
            edges = cv2.Canny(gray, 40, 120)
            edge_density = np.sum(edges > 0) / edges.size
            variance = np.var(gray)
            std_dev = np.std(gray)
            
            all_edge_densities.append(edge_density)
            all_variances.append(variance)
            all_std_devs.append(std_dev)
            
            # Show sample slots
            if i % 15 == 0:
                # Predict status with current thresholds
                is_empty = self.slot_availability_simple(slot_crop)
                status = "EMPTY" if is_empty else "OCCUPIED"
                
                print("{:<6} {:<12.4f} {:<12.2f} {:<12.2f} {:<10}".format(
                    i+1, edge_density, variance, std_dev, status))
        
        # Statistics
        print("\n" + "=" * 60)
        print("STATISTICS")
        print("=" * 60)
        print(f"Edge Density - Min: {min(all_edge_densities):.4f}, "
              f"Max: {max(all_edge_densities):.4f}, "
              f"Mean: {np.mean(all_edge_densities):.4f}")
        print(f"Variance     - Min: {min(all_variances):.2f}, "
              f"Max: {max(all_variances):.2f}, "
              f"Mean: {np.mean(all_variances):.2f}")
        print(f"Std Dev      - Min: {min(all_std_devs):.2f}, "
              f"Max: {max(all_std_devs):.2f}, "
              f"Mean: {np.mean(all_std_devs):.2f}")
        
        # Recommendations
        print("\n" + "=" * 60)
        print("CURRENT THRESHOLDS")
        print("=" * 60)
        print(f"Edge Threshold: 0.08 (occupied if > 0.08)")
        print(f"Variance Threshold: 2500 (occupied if > 2500)")
        print(f"Std Dev Threshold: 25 (used with variance)")
        
        # Suggest optimal thresholds based on distribution
        suggested_edge = np.percentile(all_edge_densities, 40)
        suggested_variance = np.percentile(all_variances, 45)
        
        print("\n" + "=" * 60)
        print("SUGGESTED THRESHOLDS (based on analysis)")
        print("=" * 60)
        print(f"Edge Threshold: {suggested_edge:.4f}")
        print(f"Variance Threshold: {suggested_variance:.2f}")
        print("\nYou can adjust these in process_video() parameters")
        
        return {
            'edge_threshold': suggested_edge,
            'variance_threshold': suggested_variance
        }
    
    # ============ VIDEO PROCESSING ============
    
    def process_video(self, output_video_path: str = None, 
                     step: int = 30,
                     edge_threshold: float = 0.08,
                     variance_threshold: float = 2500,
                     show_preview: bool = False):
        """
        Process video with improved occupancy detection
        
        Args:
            output_video_path: Path to save output video
            step: Check slots every N frames (lower = more accurate but slower)
            edge_threshold: Edge density threshold (occupied if > threshold)
            variance_threshold: Variance threshold (occupied if > threshold)
            show_preview: Show real-time preview (slower)
        """
        if not self.slots:
            raise ValueError("No parking slots loaded. Run load_slots_from_json() first")
        
        print("\n" + "=" * 60)
        print("PROCESSING VIDEO")
        print("=" * 60)
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.1f} seconds")
        print(f"\nProcessing Settings:")
        print(f"  Check interval: Every {step} frames")
        print(f"  Detection method: Simple (Edge + Variance)")
        print(f"  Edge threshold: {edge_threshold}")
        print(f"  Variance threshold: {variance_threshold}")
        print(f"  Total slots: {len(self.slots)}")
        
        if output_video_path is None:
            output_video_path = self.output_dir / "output.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        diffs = [None] * len(self.slots)
        previous_frame = None
        frame_num = 0
        stats_history = []
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate differences (detect changes)
            if frame_num % step == 0 and previous_frame is not None:
                for slot_idx, slot in enumerate(self.slots):
                    x, y, w, h = slot
                    slot_crop = frame[y:y+h, x:x+w, :]
                    diffs[slot_idx] = self.calc_diff(
                        slot_crop,
                        previous_frame[y:y+h, x:x+w, :]
                    )
            
            # Update slot status
            if frame_num % step == 0:
                if previous_frame is None:
                    # First frame - check all slots
                    arr_ = range(len(self.slots))
                else:
                    # Only check slots that changed significantly
                    max_diff = np.max(diffs) if np.max(diffs) > 0 else 1
                    arr_ = [j for j in np.argsort(diffs) if diffs[j] / max_diff > 0.4]
                
                for slot_idx in arr_:
                    slot = self.slots[slot_idx]
                    x, y, w, h = slot
                    slot_crop = frame[y:y+h, x:x+w, :]
                    
                    # Use simple detection method
                    slot_status = self.slot_availability_simple(
                        slot_crop, 
                        edge_threshold=edge_threshold,
                        variance_threshold=variance_threshold
                    )
                    
                    self.slot_status_all[slot_idx] = slot_status
            
            if frame_num % step == 0:
                previous_frame = frame.copy()
            
            # Draw bounding boxes on frame
            for slot_idx, slot in enumerate(self.slots):
                slot_status = self.slot_status_all[slot_idx]
                x, y, w, h = self.slots[slot_idx]
                
                # Green = available, Red = occupied
                color = (0, 255, 0) if slot_status else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add slot number (only show some to avoid clutter)
                if slot_idx % 20 == 0:
                    cv2.putText(frame, str(slot_idx+1), (x+5, y+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add statistics overlay
            available_count = sum(self.slot_status_all)
            total_count = len(self.slot_status_all)
            occupied_count = total_count - available_count
            occupancy_rate = (occupied_count / total_count * 100) if total_count > 0 else 0
            
            # Black background for text
            cv2.rectangle(frame, (50, 20), (700, 130), (0, 0, 0), -1)
            
            # Text with white outline for visibility
            cv2.putText(frame, f"Available: {available_count} / {total_count}", (70, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, f"Available: {available_count} / {total_count}", (70, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Occupied: {occupied_count}", (70, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(frame, f"Occupied: {occupied_count}", (70, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Save statistics
            stats_history.append({
                'frame': frame_num,
                'available': available_count,
                'occupied': occupied_count,
                'total': total_count,
                'occupancy_rate': occupancy_rate
            })
            
            # Write frame to output video
            output_video.write(frame)
            
            # Show preview if requested
            if show_preview and frame_num % 5 == 0:
                cv2.imshow('Processing Preview', cv2.resize(frame, (1280, 720)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            # Display progress
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% | Frame {frame_num}/{total_frames} | "
                      f"Available: {available_count}/{total_count}", end='\r')
            
            frame_num += 1
        
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()
        
        print(f"\n\n✓ Processed {frame_num} frames")
        print(f"✓ Output video saved to: {output_video_path}")
        
        # Save statistics to JSON
        stats_path = self.output_dir / "occupancy_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_history, f, indent=2)
        
        print(f"✓ Statistics saved to: {stats_path}")
        
        # Print summary
        if stats_history:
            avg_occupancy = np.mean([s['occupancy_rate'] for s in stats_history])
            max_occupancy = max([s['occupancy_rate'] for s in stats_history])
            min_occupancy = min([s['occupancy_rate'] for s in stats_history])
            
            print(f"\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Average Occupancy: {avg_occupancy:.1f}%")
            print(f"Max Occupancy: {max_occupancy:.1f}%")
            print(f"Min Occupancy: {min_occupancy:.1f}%")
        
        return stats_history


# ============ MAIN ============

def main():
    print("=" * 70)
    print(" " * 18 + "PARKING OCCUPANCY DETECTION")
    print(" " * 22 + "(Using Your Files)")
    print("=" * 70)
    
    detector = ParkingDetectorFromJSON(
        video_path="DJI_0012.mov",
        output_dir="parking_output"
    )
    
    # Load your parking slots JSON
    json_path = "parking_slots.json"
    
    if not Path(json_path).exists():
        print(f"\nError: {json_path} not found!")
        print("Please ensure parking_slots.json is in the current directory")
        return
    
    detector.load_slots_from_json(json_path)
    
    # Visualize loaded slots
    print("\n" + "=" * 70)
    visualize = input("Do you want to visualize loaded slots? (y/n): ").strip().lower()
    
    if visualize == 'y':
        # Try to find reference frame
        reference_frame = "parking_output/frames/frame_000000.jpg"
        if not Path(reference_frame).exists():
            # Extract first frame
            cap = cv2.VideoCapture(detector.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                Path(reference_frame).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(reference_frame, frame)
        
        detector.visualize_loaded_slots(reference_frame)
    
    # Calibrate threshold
    print("\n" + "=" * 70)
    calibrate = input("Do you want to analyze and calibrate thresholds? (recommended) (y/n): ").strip().lower()
    
    edge_threshold = 0.08  # Default
    variance_threshold = 2500  # Default
    
    if calibrate == 'y':
        reference_frame = "parking_output/frames/frame_000000.jpg"
        if not Path(reference_frame).exists():
            cap = cv2.VideoCapture(detector.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                Path(reference_frame).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(reference_frame, frame)
        
        suggested = detector.calibrate_threshold(reference_frame)
        
        if suggested:
            use_suggested = input("\nUse suggested thresholds? (y/n): ").strip().lower()
            if use_suggested == 'y':
                edge_threshold = suggested['edge_threshold']
                variance_threshold = suggested['variance_threshold']
            else:
                custom_edge = input(f"Enter edge threshold (default {edge_threshold}): ").strip()
                if custom_edge:
                    edge_threshold = float(custom_edge)
                
                custom_var = input(f"Enter variance threshold (default {variance_threshold}): ").strip()
                if custom_var:
                    variance_threshold = float(custom_var)
    
    # Process video
    print("\n" + "=" * 70)
    proceed = input("Process video now? (y/n): ").strip().lower()
    
    if proceed == 'y':
        step = input("Check slots every N frames (default 30, lower=slower but accurate): ").strip()
        step = int(step) if step else 30
        
        preview = input("Show real-time preview? (y/n, slower if yes): ").strip().lower()
        show_preview = (preview == 'y')
        
        print(f"\nStarting video processing...")
        print(f"Using SIMPLE detection method")
        print(f"  Edge threshold: {edge_threshold}")
        print(f"  Variance threshold: {variance_threshold}")
        
        detector.process_video(
            step=step,
            edge_threshold=edge_threshold,
            variance_threshold=variance_threshold,
            show_preview=show_preview
        )
        
        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nResults saved in: {detector.output_dir}/")
        print(f"  • Output video: output.mp4")
        print(f"  • Statistics: occupancy_stats.json")
    else:
        print("\nYou can run the processing later by calling:")
        print(f"  detector.process_video(step=30, edge_threshold={edge_threshold}, variance_threshold={variance_threshold})")


if __name__ == "__main__":
    main()