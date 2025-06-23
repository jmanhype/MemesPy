"""Comprehensive metadata collection service for meme generation."""

import os
import time
import json
import hashlib
import platform
import psutil
import socket
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import base64
from io import BytesIO

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import piexif
    HAS_PIEXIF = True
except ImportError:
    HAS_PIEXIF = False

try:
    import numpy as np
    from colorthief import ColorThief
    HAS_ANALYSIS_LIBS = True
except ImportError:
    HAS_ANALYSIS_LIBS = False

HAS_IMAGE_LIBS = HAS_PIL and HAS_ANALYSIS_LIBS

import logging

logger = logging.getLogger(__name__)


class MetadataCollector:
    """Collects comprehensive metadata during meme generation."""
    
    def __init__(self):
        """Initialize the metadata collector."""
        self.start_time = None
        self.metadata = {}
        self.generation_id = None
        
    def start_generation(self, topic: str, format: str, **kwargs) -> str:
        """Start tracking a new generation."""
        self.start_time = time.time()
        self.generation_id = hashlib.md5(
            f"{topic}:{format}:{time.time()}".encode()
        ).hexdigest()
        
        # Initialize metadata structure
        self.metadata = {
            'generation_id': self.generation_id,
            'topic': topic,
            'format': format,
            'started_at': datetime.utcnow().isoformat(),
            'system_metadata': self._collect_system_metadata(),
            'request_metadata': {
                'topic': topic,
                'format': format,
                'additional_params': kwargs
            },
            'generation_steps': []
        }
        
        return self.generation_id
    
    def _collect_system_metadata(self) -> Dict[str, Any]:
        """Collect system-level metadata."""
        try:
            return {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
                'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2),
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'process_id': os.getpid(),
                'environment': os.getenv('ENVIRONMENT', 'development')
            }
        except Exception as e:
            logger.warning(f"Failed to collect system metadata: {e}")
            return {}
    
    def track_dspy_generation(self, 
                            predictor_class: str,
                            inputs: Dict[str, Any],
                            outputs: Dict[str, Any],
                            duration_ms: float,
                            model_info: Dict[str, Any]) -> None:
        """Track DSPy text generation metadata."""
        step_metadata = {
            'step_type': 'dspy_generation',
            'timestamp': datetime.utcnow().isoformat(),
            'duration_ms': duration_ms,
            'predictor_class': predictor_class,
            'inputs': inputs,
            'outputs': outputs,
            'model_info': model_info,
            'token_usage': self._extract_token_usage(outputs),
            'quality_metrics': self._analyze_text_quality(outputs.get('text', ''))
        }
        
        self.metadata['generation_steps'].append(step_metadata)
        
        # Update main metadata
        if 'dspy_metadata' not in self.metadata:
            self.metadata['dspy_metadata'] = {}
        
        self.metadata['dspy_metadata'].update({
            'predictor_class': predictor_class,
            'model_name': model_info.get('model'),
            'temperature': model_info.get('temperature'),
            'total_tokens': step_metadata['token_usage'].get('total_tokens', 0)
        })
    
    def track_image_generation(self,
                             provider: str,
                             prompt: str,
                             model: str,
                             size: str,
                             duration_ms: float,
                             success: bool,
                             image_url: Optional[str] = None,
                             error: Optional[str] = None,
                             fallback_info: Optional[Dict] = None) -> None:
        """Track image generation metadata."""
        step_metadata = {
            'step_type': 'image_generation',
            'timestamp': datetime.utcnow().isoformat(),
            'duration_ms': duration_ms,
            'provider': provider,
            'model': model,
            'prompt': prompt,
            'size': size,
            'success': success,
            'image_url': image_url,
            'error': error,
            'fallback_info': fallback_info
        }
        
        self.metadata['generation_steps'].append(step_metadata)
        
        # Update main metadata
        if 'image_generation' not in self.metadata:
            self.metadata['image_generation'] = {}
        
        self.metadata['image_generation'].update({
            'final_provider': provider,
            'final_model': model,
            'generation_time_ms': duration_ms,
            'used_fallback': fallback_info is not None,
            'fallback_reason': fallback_info.get('reason') if fallback_info else None
        })
    
    def analyze_generated_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze generated image for metadata."""
        if not HAS_IMAGE_LIBS:
            logger.warning("Image analysis libraries not available")
            return {}
        
        try:
            image_metadata = {}
            
            # Basic file info
            path = Path(image_path)
            if path.exists():
                stat = path.stat()
                image_metadata['file_size_bytes'] = stat.st_size
                image_metadata['file_size_mb'] = round(stat.st_size / (1024**2), 2)
            
            # Open image for analysis
            with Image.open(image_path) as img:
                # Basic properties
                image_metadata['format'] = img.format
                image_metadata['mode'] = img.mode
                image_metadata['width'] = img.width
                image_metadata['height'] = img.height
                image_metadata['aspect_ratio'] = round(img.width / img.height, 2)
                
                # Color analysis
                image_metadata['has_transparency'] = img.mode in ('RGBA', 'LA', 'PA')
                
                # Get dominant colors
                if HAS_IMAGE_LIBS:
                    try:
                        # Save to BytesIO for ColorThief
                        buffer = BytesIO()
                        img.save(buffer, format='PNG')
                        buffer.seek(0)
                        
                        color_thief = ColorThief(buffer)
                        dominant_color = color_thief.get_color(quality=1)
                        palette = color_thief.get_palette(color_count=5, quality=1)
                        
                        image_metadata['dominant_color'] = {
                            'rgb': dominant_color,
                            'hex': '#{:02x}{:02x}{:02x}'.format(*dominant_color)
                        }
                        image_metadata['color_palette'] = [
                            {'rgb': color, 'hex': '#{:02x}{:02x}{:02x}'.format(*color)}
                            for color in palette
                        ]
                    except Exception as e:
                        logger.warning(f"Color analysis failed: {e}")
                
                # Image statistics
                if img.mode in ('RGB', 'RGBA'):
                    img_array = np.array(img)
                    image_metadata['brightness'] = float(np.mean(img_array))
                    image_metadata['contrast'] = float(np.std(img_array))
                
                # EXIF data (if any)
                exif_data = img.getexif()
                if exif_data:
                    image_metadata['has_exif'] = True
                    image_metadata['exif_tags'] = {
                        TAGS.get(k, k): v 
                        for k, v in exif_data.items()
                    }
                else:
                    image_metadata['has_exif'] = False
            
            return image_metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze image: {e}")
            return {'error': str(e)}
    
    def _extract_token_usage(self, outputs: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from model outputs."""
        # This would need to be implemented based on actual DSPy output format
        return {
            'prompt_tokens': outputs.get('prompt_tokens', 0),
            'completion_tokens': outputs.get('completion_tokens', 0),
            'total_tokens': outputs.get('total_tokens', 0)
        }
    
    def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze text quality metrics."""
        if not text:
            return {}
        
        words = text.split()
        sentences = text.split('.')
        
        return {
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': round(sum(len(w) for w in words) / max(1, len(words)), 2),
            'has_emojis': any(ord(c) > 127 for c in text),
            'has_hashtags': '#' in text,
            'capitalization_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text))
        }
    
    def add_cost_tracking(self, 
                         text_cost: float = 0.0,
                         image_cost: float = 0.0,
                         storage_cost: float = 0.0) -> None:
        """Add cost tracking metadata."""
        if 'cost_metadata' not in self.metadata:
            self.metadata['cost_metadata'] = {
                'text_generation_cost': 0.0,
                'image_generation_cost': 0.0,
                'storage_cost': 0.0,
                'total_cost': 0.0
            }
        
        self.metadata['cost_metadata']['text_generation_cost'] += text_cost
        self.metadata['cost_metadata']['image_generation_cost'] += image_cost
        self.metadata['cost_metadata']['storage_cost'] += storage_cost
        self.metadata['cost_metadata']['total_cost'] = sum(
            self.metadata['cost_metadata'].values()
        ) - self.metadata['cost_metadata']['total_cost']
    
    def add_moderation_result(self, 
                            status: str,
                            scores: Dict[str, float],
                            flagged_categories: List[str] = None) -> None:
        """Add content moderation metadata."""
        self.metadata['moderation_metadata'] = {
            'moderation_status': status,
            'moderation_scores': scores,
            'flagged_categories': flagged_categories or [],
            'moderated_at': datetime.utcnow().isoformat()
        }
    
    def finalize(self, 
                score: float,
                image_url: str,
                meme_text: str,
                success: bool = True,
                error: Optional[str] = None) -> Dict[str, Any]:
        """Finalize metadata collection."""
        end_time = time.time()
        total_duration = (end_time - self.start_time) * 1000 if self.start_time else 0
        
        # Core completion data
        self.metadata.update({
            'completed_at': datetime.utcnow().isoformat(),
            'total_duration_ms': round(total_duration, 2),
            'success': success,
            'error': error,
            'score': score,
            'image_url': image_url,
            'meme_text': meme_text
        })
        
        # Calculate generation efficiency
        retry_count = sum(
            1 for step in self.metadata.get('generation_steps', [])
            if step.get('step_type') == 'image_generation' and not step.get('success')
        )
        
        self.metadata['generation_metadata'] = {
            'total_duration_ms': round(total_duration, 2),
            'retry_count': retry_count,
            'steps_count': len(self.metadata.get('generation_steps', [])),
            'efficiency_score': self._calculate_efficiency(total_duration, retry_count)
        }
        
        return self.metadata
    
    def _calculate_efficiency(self, duration_ms: float, retry_count: int) -> float:
        """Calculate generation efficiency score."""
        time_score = max(0, 1 - (duration_ms / 10000))
        retry_penalty = 1 / (1 + retry_count)
        return round(time_score * 0.7 + retry_penalty * 0.3, 3)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps(self.metadata, indent=2, default=str)
    
    def embed_in_image_exif(self, image_path: str, output_path: Optional[str] = None) -> bool:
        """Embed metadata in image EXIF data."""
        if not HAS_PIEXIF:
            logger.warning("piexif not available for EXIF embedding")
            return False
        
        if not HAS_PIL:
            logger.warning("PIL not available for EXIF embedding")
            return False
        
        try:
            # Create EXIF data
            exif_dict = {
                "0th": {
                    piexif.ImageIFD.Software: "DSPy Meme Generator",
                    piexif.ImageIFD.DateTime: datetime.utcnow().strftime("%Y:%m:%d %H:%M:%S"),
                    piexif.ImageIFD.HostComputer: platform.node(),
                },
                "Exif": {
                    piexif.ExifIFD.DateTimeOriginal: datetime.utcnow().strftime("%Y:%m:%d %H:%M:%S"),
                    piexif.ExifIFD.UserComment: json.dumps({
                        'generation_id': self.generation_id,
                        'topic': self.metadata.get('topic'),
                        'format': self.metadata.get('format'),
                        'score': self.metadata.get('score'),
                        'model': self.metadata.get('image_generation', {}).get('final_model'),
                        'provider': self.metadata.get('image_generation', {}).get('final_provider'),
                        'generation_time_ms': self.metadata.get('generation_metadata', {}).get('total_duration_ms')
                    }).encode('utf-8')
                }
            }
            
            # Convert to bytes
            exif_bytes = piexif.dump(exif_dict)
            
            # Open image and save with EXIF
            img = Image.open(image_path)
            output = output_path or image_path
            img.save(output, exif=exif_bytes)
            
            logger.info(f"Successfully embedded EXIF metadata in {output}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to embed EXIF data: {e}")
            return False


class MetadataAggregator:
    """Aggregates metadata for analytics and reporting."""
    
    def __init__(self):
        """Initialize the aggregator."""
        self.metrics = []
        
    def add_generation(self, metadata: Dict[str, Any]) -> None:
        """Add a generation's metadata to aggregation."""
        self.metrics.append(metadata)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get aggregated performance metrics."""
        if not self.metrics:
            return {}
        
        durations = [m['total_duration_ms'] for m in self.metrics if 'total_duration_ms' in m]
        scores = [m['score'] for m in self.metrics if 'score' in m]
        
        return {
            'total_generations': len(self.metrics),
            'successful_generations': sum(1 for m in self.metrics if m.get('success', False)),
            'avg_duration_ms': round(sum(durations) / len(durations), 2) if durations else 0,
            'min_duration_ms': min(durations) if durations else 0,
            'max_duration_ms': max(durations) if durations else 0,
            'avg_score': round(sum(scores) / len(scores), 2) if scores else 0,
            'model_usage': self._count_model_usage(),
            'error_rate': self._calculate_error_rate()
        }
    
    def _count_model_usage(self) -> Dict[str, int]:
        """Count usage by model."""
        usage = {}
        for m in self.metrics:
            model = m.get('image_generation', {}).get('final_model', 'unknown')
            usage[model] = usage.get(model, 0) + 1
        return usage
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate."""
        if not self.metrics:
            return 0.0
        errors = sum(1 for m in self.metrics if not m.get('success', True))
        return round(errors / len(self.metrics), 3)