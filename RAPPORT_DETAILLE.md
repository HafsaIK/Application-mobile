# Rapport Détaillé — Système d'Assistance Visuelle pour Personnes Malvoyantes

**Version** : 1.0
**Langue** : Français
**Fichier** : `RAPPORT_DETAILLE.md`

---

**Résumé**
Ce document présente, de façon détaillée mais sans inclure de code ni de spécifications d'entraînement, un système d'assistance visuelle destiné aux personnes malvoyantes. Le système combine détection d'objets, OCR multilingue et synthèse vocale via une architecture client-serveur. L'objectif est d'offrir des descriptions utiles et vocalisées de l'environnement, ainsi qu'une lecture de textes en arabe et en anglais, avec une application mobile cross-platform pour l'usage quotidien.

---

**1. Contexte et objectifs**
- Contexte : améliorer l'autonomie des personnes malvoyantes en exploitant des modèles de vision et d'OCR.
- Objectifs : fournir détection d'objets fiable, reconnaissance de texte multilingue (Arabe/Anglais) et synthèse vocale claire; maintenir latence acceptable pour interactions humaines; déployer une application accessible sur mobiles/desktop.

---

**2. Architecture système (vue d'ensemble)**
- Client mobile (Flutter) : capture image / flux caméra, gère permissions, interface accessible (contrôles, lecture audio, navigation vocale).
- Serveur d'inférence (REST API) : reçoit images, exécute pipeline de détection + OCR, génère phrases descriptives et fichiers audio, renvoie image annotée et texte.
- Modules principaux au serveur : gestion des modèles, pipeline de fusion des détections, pipeline OCR multilingue, moteur TTS (cloud + fallback local).
- Flux : capture → upload → inférence (détection + OCR) → génération de phrase et audio → réponse au client.

---

**3. Principes méthodologiques (sans détails d'entraînement)**
- Détection hybride : on combine un modèle spécialisé (pour les classes les plus pertinentes pour l'utilisateur) et un modèle générique pour élargir la couverture des objets. Les prédictions sont fusionnées selon une règle simple : priorité aux classes couvertes par le modèle spécialisé, complétées par les prédictions génériques pour les autres classes.

- OCR multilingue : pipeline en deux étapes — détection des régions de texte puis reconnaissance dirigée par script/une heuristique de langue. Les régions contenant des caractères arabes sont traitées par la chaîne spécialisée pour l'arabe ; les régions latines par le module optimisé pour l'anglais. Le résultat est trié (RTL pour l'arabe, LTR pour l'anglais) avant restitution.

- Synthèse vocale : génération d'une phrase descriptive (ex. « Attention, dans cette image il y a : personne, voiture, tasse. ») en français pour la synthèse principale, tandis que le texte détecté est restitué dans sa langue d'origine (arabe/anglais). Stratégie TTS : priorité à une synthèse de haute qualité (cloud) avec un mécanisme de secours offline pour situations réseau dégradé.

---

**4. Données et protocoles d'évaluation (concepts)**
- Datasets : images d'environnements domestiques et urbains pour la détection ; collections de documents et panneaux locaux pour l'OCR (imprimé et manuscrit). Les jeux de test sont séparés des données utilisées pour toute optimisation.
- Métriques : détection (mAP@0.5, précision, rappel, F1) ; OCR (CER, WER) ; performance opérationnelle (latence end-to-end, débit) ; acceptabilité utilisateur (SUS, taux de réussite des tâches).
- Étude utilisateur : sessions contrôlées avec participants malvoyants, tâches réalistes (identification d'objets, lecture de documents, évaluation de scène) et questionnaire SUS.

---

**5. Résultats clés**
- Détection : la fusion des deux modèles améliore la précision globale et l'équilibre précision/rappel par rapport à l'utilisation d'un seul modèle ; F1 observé autour de 0,80 dans les conditions testées.
- OCR : performance élevée sur texte imprimé ; l'arabe (contexte marocain) est correctement reconnu par la chaîne spécialisée ; l'écriture manuscrite reste le cas le plus difficile, avec taux d'erreur plus élevé.
- Latence : end-to-end typiquement inférieure à 1,1 seconde sur infrastructure avec accélération GPU et réseau local ; la synthèse vocale cloud et la transmission réseau sont les principaux contributeurs à la latence.
- Usabilité : SUS moyen ≈ 76.8/100 ; participants apprécient la qualité des annonces et la fonctionnalité de lecture de documents ; demandes fréquentes pour descriptions spatiales et réduction de la latence.

---

**6. Analyse détaillée**
- Bénéfices de la fusion : couverture d'objets plus large, réduction des faux positifs pour classes spécialisées, simplicité d'implémentation et interprétabilité.
- Limitations constatées : sensibilité en faible luminosité, difficulté sur petits objets ou très occlus, variabilité sur écriture manuscrite et textes fortement dégradés.
- Robustesse : le pipeline inclut seuils de confiance et déduplication pour limiter annonces erronées ; toutefois, il faut gérer la communication de l'incertitude à l'utilisateur (ex. préfixer par "probablement" ou fournir scores si demandé).

---

**7. Cas d'usage et scénarios d'exploitation**
- Usage quotidien domestique : identifier objets proches (tasse, porte, chaise), lecture d'étiquettes et petites notices.
- Mobilité assistée (non critique) : description de la scène, repérage d'obstacles visuels non dynamiques ; ne remplace pas les aides physiques pour navigation critique.
- Lecture de documents et affiches : aide à la compréhension des documents imprimés et panneaux.
- Accessibilité administrative : lecture d'informations écrites (étiquettes, factures) dans la langue d'origine.

---

**8. Déploiement et exploitation**
- Options d'hébergement : serveur local (edge) pour latence faible et vie privée ; serveur cloud pour évolutivité et puissance de calcul centralisée.
- Maintenance : surveillance des modèles (dérive de performance), pipeline de logs anonymisés pour améliorer modèles et UX, mécanisme de mises à jour testées graduellement.
- Confidentialité : traiter les images localement si possible, anonymisation et consentement explicite pour données partagées, chiffrement des communications.

---

**9. Recommandations pratiques**
- Prioriser déploiement edge pour utilisateurs mobiles avec contrainte réseau ; solution cloud hybride pour opérations intensives.
- Ajouter retour spatial simple ("à gauche", "au centre", "à droite") via heuristiques de positionnement à court terme.
- Mettre en place un mécanisme de remontée d'erreurs utilisateurs pour enrichir dataset et améliorer reconnaissance manuscrite.
- Proposer réglages utilisateur (seuil de confiance, fréquence d'annonces, langue de synthèse).

---

**10. Travaux futurs (priorités)**
- Réduction de latence TTS : intégrer moteur TTS local de haute qualité ou optimiser pipeline cloud.
- Amélioration OCR manuscrit : collecte ciblée de données manuscrites et post-traitement linguistique.
- Personnalisation : adaptation utilisateur (préférences, objets d'intérêt) via apprentissage few-shot.
- Fonctionnalités avancées : estimation de distance, détection d'actions, intégration à assistants vocaux.

---

**11. Conclusion**
Ce rapport détaillé présente une solution modulaire, pragmatique et centrée utilisateur pour l'assistance visuelle. Sans entrer dans les détails de code ou d'entraînement, il montre que la combinaison d'une détection hybride, d'un OCR multilingue et d'une synthèse vocale soigneuse permet d'obtenir une solution utile et acceptable pour des usages quotidiens des personnes malvoyantes. Les axes d'amélioration identifiés permettront d'augmenter la robustesse et la couverture fonctionnelle lors des prochaines itérations.

---

**Références sélectionnées**
- Publications et ressources standard sur YOLO, TrOCR, PaddleOCR, FastAPI et Flutter (références complètes disponibles dans les rapports existants du projet).

---

Fin du document. Si vous souhaitez une version prête à imprimer (PDF), une traduction vers l'anglais, ou une adaptation pour une présentation (slides), je peux la générer et l'enregistrer dans le projet.
