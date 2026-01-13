"""
CacheManager - Gestion avancée du cache pour GenAI
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import hashlib

class CacheManager:
    """Gestionnaire de cache avec expiration et compression"""
    
    def __init__(self, cache_dir: str = "cache/genai", ttl_days: int = 7):
        self.cache_dir = cache_dir
        self.ttl_days = ttl_days
        os.makedirs(cache_dir, exist_ok=True)
        
        # Nettoyer le cache expiré au démarrage
        self._clean_expired_cache()
    
    def get(self, key: str) -> Any:
        """Récupère une valeur du cache"""
        filepath = self._get_filepath(key)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Vérifier l'expiration
            if self._is_expired(data.get('timestamp')):
                os.remove(filepath)
                return None
            
            return data.get('data')
            
        except Exception:
            return None
    
    def set(self, key: str, value: Any):
        """Stocke une valeur dans le cache"""
        filepath = self._get_filepath(key)
        
        cache_data = {
            'data': value,
            'timestamp': datetime.now().isoformat(),
            'key': key
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur écriture cache: {e}")
    
    def generate_key(self, prompt: str, context: Dict) -> str:
        """Génère une clé unique basée sur le prompt et contexte"""
        # Combiner prompt et contexte
        content = prompt + json.dumps(context, sort_keys=True)
        
        # Hash pour clé fixe
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_filepath(self, key: str) -> str:
        """Génère le chemin du fichier cache"""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def _is_expired(self, timestamp: str) -> bool:
        """Vérifie si une entrée cache est expirée"""
        if not timestamp:
            return True
        
        try:
            cache_time = datetime.fromisoformat(timestamp)
            expiry_time = cache_time + timedelta(days=self.ttl_days)
            return datetime.now() > expiry_time
        except Exception:
            return True
    
    def _clean_expired_cache(self):
        """Nettoie les entrées cache expirées"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.cache_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if self._is_expired(data.get('timestamp')):
                        os.remove(filepath)
                        
                except Exception:
                    os.remove(filepath)
    
    def clear_all(self):
        """Vide tout le cache"""
        import shutil
        shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir)