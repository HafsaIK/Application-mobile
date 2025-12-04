# Rapport Centré sur l'Application — Assistant Malvoyant

**Fichier** : `RAPPORT_APPLICATION.md`
**Objectif** : Décrire en détail la partie application (client) du projet : architecture, UX, intégration avec le backend, accessibilité, tests, déploiement et bonnes pratiques.

---

## 1. Vue d'ensemble de l'application

L'application est une application Flutter cross-platform destinée à fournir, via une interface accessible, une assistance visuelle aux personnes malvoyantes. Elle capture des images ou un flux caméra, envoie des requêtes au serveur d'inférence, reçoit des résultats annotés et du texte, puis les fournit à l'utilisateur sous forme audio et visuelle (pour accompagnateurs).

Objectifs fonctionnels principaux :
- Capture d'image / flux caméra
- Envoi sécurisé d'images au backend
- Réception et affichage des résultats (image annotée, texte)
- Lecture audio (TTS) des descriptions et du texte détecté
- Contrôles d'accessibilité et paramètres utilisateur

---

## 2. Architecture client

Composants majeurs :
- UI (Flutter) : écrans principaux, navigation
- Service HTTP : `api_service` qui gère les appels au backend, les retries et timeouts
- Gestion média : modules pour capture image, gestion caméra, sélection depuis galerie
- Audio : gestion lecture TTS locale et contrôle audio (pause, replay)
- Stockage local : cache des derniers résultats, fichiers audio mis en cache
- Permissions et diagnostics : gestion runtime des permissions caméra et stockage

Architecture logique :
- `UI <-> api_service <-> réseau` ; audio et stockage accessibles depuis UI
- Mode offline/backup : si le serveur n'est pas disponible, l'app propose lecture locale (fallback TTS) et enfile les requêtes pour synchronisation ultérieure si demandé.

---

## 3. Écrans et flux utilisateur

### 3.1 Écran d'accueil / démarrage
- Vérifie les permissions (caméra, micro, stockage) à l'initialisation
- Indique état de connexion au serveur (en ligne/hors ligne)
- Boutons : "Détection en direct", "Choisir une image", "Paramètres", "Aide"

### 3.2 LiveDetectionScreen
- Flux caméra en plein écran
- Bouton capture (photo) et mode snapshot automatique (ex : capture toutes les N secondes si activé)
- Indicateur d'activité réseau et latence estimée
- Option d'activer/désactiver envoi continu vs capture manuelle
- Retour haptique et audio bref (ex. "Image envoyée")

Flux : capture → vignette de confirmation → envoi → attente → lecture audio automatique + affichage résultat

### 3.3 ResultScreen
- Affiche l'image annotée renvoyée par le serveur (pour accompagnateur)
- Transcription du texte détecté (séparée par langue)
- Contrôles audio : lecture / pause / relire phrase / télécharger audio
- Boutons d'actions : renvoyer l'image, partager, marquer comme confiance faible

### 3.4 Paramètres
- Langue de synthèse (français par défaut), volume, voix préférée
- Seuil de confiance : régler sensibilité des annonces
- Mode offline : activer TTS local (pyttsx3) quand indisponible
- Options de confidentialité : suppression automatique des images, envoi anonymisé

### 3.5 Aide & tutoriel accessibilité
- Tutoriels vocaux guidés (onboarding) décrivant usage
- Raccourcis (gestes) pour utilisateurs non voyants

---

## 4. Intégration Backend (API)

Principes d'intégration :
- Communication via HTTPS multipart/form-data pour l'envoi d'images
- Endpoints principaux : `/detect` (image → description + image annotée), `/ocr` (image → texte), `/health`
- Gestion des erreurs réseau : timeouts, retries avec backoff, messages d'erreur vocaux compréhensibles
- Sécurité : TLS obligatoire, tokens d'authentification optionnels pour déploiements privés

Bonnes pratiques côté client :
- Envoyer images encodées en JPEG (quality configurable) pour réduire latence
- Limiter taille des images et proposer redimensionnement progressif (ex : 640×640) avant upload
- Gérer l'état de connexion et proposer file d'attente synchronisée

---

## 5. Accessibilité (a11y)

Conception orientée utilisateurs malvoyants :
- Interface navigable par lecteurs d'écran (semantics labels sur tous les éléments)
- Feedback vocal constant : confirmations, erreurs, conseils
- Contraste élevé pour composants visuels (pour accompagnateurs ou utilisateurs à basse vision)
- Touch targets larges (>48dp), gestes simples et navigation linéaire
- Mode simplifié (moins de choix à l'écran) pour sessions rapides
- Personnalisation : vitesse de lecture, langage TTS, fréquence d'annonces, mode silencieux

Exemples d'implémentation :
- Chaque bouton a un label vocal explicite (ex. `Semantics(label: "Prendre une photo")`)
- Notifications sonores/optiques pour états critiques (ex. "Pas de caméra détectée")

---

## 6. Gestion audio et TTS

Stratégies TTS :
- Priorité au TTS cloud (qualité voix) avec fallback local
- Cache audio : stocker mp3/wav des phrases souvent prononcées pour réemploi
- Contrôle de latence : jouer un message court localement pendant que la version cloud arrive (ex. "Analyse en cours")
- Préférences : choix de voix, vitesse, langue de synthèse

Sécurité et confidentialité :
- S'assurer que les fichiers audio stockés sont chiffrés ou effacés automatiquement selon paramètre

---

## 7. Performance et optimisation mobile

Points d'attention :
- Reduire taille de l'image (compression) avant envoi pour diminuer latence
- Batch/Throttle : dans le mode streaming, contrôler fréquence d'envoi (ex. 1 fps) et ne pas saturer réseau
- Utiliser cache pour éviter renvois répétés sur la même scène
- Mesures métriques embarquées : temps d'envoi, temps d'inférence, taux d'erreurs

Optimisations avancées :
- Basculer vers inférence on-device (via TFLite/ONNX) pour certains scénarios afin de diminuer latence
- Réduction de la consommation d'énergie : réduire résolution du flux quand l'app en arrière-plan

---

## 8. Gestion des permissions & diagnostics

- Demander permissions au moment juste (ex. avant accès caméra)
- Fournir messages clairs sur pourquoi la permission est requise
- Offrir un écran de diagnostics (vérification caméra, réseau, micro)
- Log local minimal et anonymisé pour retours bugs (avec opt-in utilisateur)

---

## 9. Tests et assurance qualité

Tests recommandés :
- Unitaires : test des services API, parsing des réponses
- Intégration : cycle complet capture→upload→réponse→lecture audio
- Tests d'accessibilité : vérifier labels semantics, navigation via lecteur d'écran
- Tests de performance : mesurer latence en scénarios réseau variés
- Tests utilisateurs : séances de validation avec plusieurs profils de déficience visuelle

Automatisation :
- CI pipelines pour builds Flutter (Android/iOS) et tests unitaires
- Test matrix : devices réels et émulateurs (Android API levels, iOS versions)

---

## 10. Déploiement et builds

Commandes usuelles (exemples) :

```powershell
# Installer dépendances Flutter
cd mobile_app
flutter pub get

# Lancer sur appareil connecté
flutter run

# Build release Android
flutter build apk --release

# Build release Windows
flutter build windows --release
```

Recommandations :
- Utiliser builds signés pour publication
- Configurer processus de distribution (Play Store, App Store) avec politique de confidentialité
- Fournir mise à jour OTA (via stores) et canaux beta pour tests utilisateurs

---

## 11. Journalisation et monitorage

- Collecte métriques anonymes : latence, erreurs, taux de réussite
- Dashboards pour suivre santé de l'app et service backend
- Alertes sur taux d'erreur élevé ou latence moyenne dépassée

---

## 12. Confidentialité et conformité

- Minimiser conservation d'images (suppression automatique configurable)
- Consentement explicite pour toute collecte
- Chiffrement en transit (TLS) et au repos (optionnel pour cache local)
- Documenter politique de confidentialité et options d'opt-out

---

## 13. Roadmap d'amélioration de l'application

Priorités immédiates :
- Ajouter retour spatial (gauche / centre / droite)
- Mécanisme d'ajustement de sensibilité (seuils de confiance)
- Cache audio et phrases fréquemment utilisées

Moyen terme :
- On-device inference partiel
- Personnalisation (objets favoris, réglages utilisateurs)
- Mode "basse consommation" pour longue autonomie

Long terme :
- Intégration avec assistants vocaux et wearables
- Support multi-utilisateurs et profils

---

## 14. Checklist pour déploiement produit

- [ ] Tests d'accessibilité complétés
- [ ] Politique de confidentialité rédigée et accessible
- [ ] Build release signé pour plateformes cibles
- [ ] Processus de feedback utilisateur en place
- [ ] Processus de CI/CD configuré pour releases

---

## Conclusion

Ce rapport focalisé sur l'application décrit en détail comment l'interface client doit fonctionner, communiquer avec le backend et offrir une expérience accessible et robuste aux personnes malvoyantes. Il sert de guide pour l'équipe produit et développement afin d'optimiser l'usabilité, la performance et la confidentialité lors du déploiement.

---

*Fichier généré automatiquement par l'assistant — si vous souhaitez des sections additionnelles (maquettes d'écran, arbre de composants Flutter, spécifications API détaillées), je peux les produire maintenant.*