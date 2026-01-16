# 🎓 Assistant Enseignant Intelligent (Intelligent Teacher Assistant)

Un agent IA avancé conçu pour assister les enseignants dans la gestion des questions des étudiants et la fourniture de ressources pédagogiques pertinentes.

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Technologies utilisées](#technologies-utilisées)
- [Auteurs](#auteurs)

## 🎯 Vue d'ensemble

L'Assistant Enseignant Intelligent est une application basée sur LangChain et Streamlit qui combine :
- **Un modèle de langage conversationnel** pour générer des réponses adaptées
- **Une base de données vectorielle** pour retrouver les documents pertinents
- **Une mémoire conversationnelle** pour maintenir le contexte des discussions
- **Une interface web intuitive** pour faciliter l'interaction

Cette application est particulièrement adaptée pour :
- Répondre aux questions des étudiants sur les matériaux du cours
- Fournir des explications et des clarifications
- Maintenir un contexte de conversation cohérent

## ✨ Fonctionnalités

- **Chat conversationnel** : Interactions naturelles et fluides avec l'IA
- **Récupération de documents** : Trouve automatiquement les ressources pertinentes dans la base de données
- **Mémoire contextuelle** : Conserve l'historique de la conversation pour des réponses plus cohérentes
- **Support multi-documents** : Traite des fichiers PDF provenant du répertoire `data/`
- **Chunking intelligent** : Division des documents en segments optimisés pour la recherche
- **Embeddings performants** : Utilise des modèles HuggingFace pour la vectorisation

## 🏗️ Architecture

Le projet suit une architecture modulaire :

```
intelligent-teacher-assistant/
├── app.py                      # Application Streamlit principale
├── src/
│   ├── config.py              # Configuration globale
│   ├── prompt_template.py     # Templates des prompts
│   └── teacher_assistant/
│       ├── vectorial_db.py    # Gestion de la base vectorielle
│       ├── retriever.py       # Récupération et chaînes LLM
│       └── memory_utils.py    # Gestion de la mémoire conversationnelle
├── data/                      # Dossier contenant les fichiers PDF
└── chroma_db/                # Base de données vectorielle Chroma
```

### Flux de traitement

1. **Chargement des données** : Les PDF du dossier `data/` sont chargés
2. **Chunking** : Les documents sont divisés en segments chevauchants
3. **Embedding** : Les segments sont vectorisés avec le modèle `sentence-transformers`
4. **Stockage** : Les vecteurs sont stockés dans Chroma
5. **Recherche** : À chaque question, les documents pertinents sont retrouvés
6. **Réponse** : Le LLM génère une réponse basée sur les documents et l'historique

## 📦 Prérequis

- Python 3.11+
- pip ou Poetry
- Une clé API OpenAI (pour GPT-4o mini)

## 🚀 Installation

### Avec Poetry

```bash
# Cloner le repository
git clone <repository_url>
cd intelligent-teacher-assistant

# Installer les dépendances
poetry install
poetry build
poetry shell
```


## ⚙️ Configuration

### Variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
OPENAI_API_KEY=votre_clé_api_openai
```

### Fichiers de configuration

Modifier `src/config.py` pour ajuster :

- `CHUNK_SIZE` : Taille des segments (défaut: 1000)
- `CHUNK_OVERLAP` : Chevauchement entre segments (défaut: 200)
- `EMBEDDING_MODEL` : Modèle d'embedding (défaut: `sentence-transformers/all-MiniLM-L6-v2`)
- `TEMPERATURE` : Paramètre de créativité du LLM (défaut: 1.0)

### Organisation des données

Placer les fichiers PDF dans le dossier `data/` :

```
data/
├── cours_1.pdf
├── cours_2.pdf
└── ressources.pdf
```

## 💻 Utilisation

### Lancer l'application

```bash
# Avec Poetry
poetry run streamlit run app.py

# Avec Python standard
streamlit run app.py
```

L'application s'ouvrira dans votre navigateur à `http://localhost:8501`

### Interface utilisateur

1. **Sidebar** : Configuration de l'identifiant étudiant et paramètres
2. **Zone principale** : Chat conversationnel
3. **Historique** : Conservé pendant la session (optionnel)

## 📁 Structure du projet

```
src/
├── config.py
│   └── Configuration centralisée (chemins, paramètres)
│
├── prompt_template.py
│   └── Templates des prompts personnalisés
│
└── teacher_assistant/
    ├── vectorial_db.py
    │   ├── data_loading()         # Charge les PDF
    │   ├── text_chunking()        # Divise les documents
    │   ├── embedding_initialization() # Initialise les embeddings
    │   └── create_vector_db()     # Crée la base Chroma
    │
    ├── retriever.py
    │   ├── init_llm()             # Initialise le modèle GPT
    │   ├── retriever()            # Crée une chaîne QA
    │   └── prof_assistant()       # Crée une chaîne conversationnelle
    │
    └── memory_utils.py
        └── Utilitaires pour la gestion de la mémoire
```

## 🛠️ Technologies utilisées

| Technologie | Version | Rôle |
|---|---|---|
| Python | 3.11+ | Langage principal |
| LangChain | 0.3.7+ | Framework pour les chaînes IA |
| Streamlit | - | Interface web |
| OpenAI API | - | Modèle GPT-4o mini |
| Chroma | - | Base de données vectorielle |
| HuggingFace | 4.46.2+ | Modèles d'embedding |
| Sentence Transformers | - | Modèles d'embedding |
| Gradio | 6.2.0 | Interface alternative (optionnel) |

## 📋 Dépendances principales

```toml
python = "^3.11"
python-dotenv = "^1.0.1"
langchain = "^0.3.7"
langchain-community = "^0.3.7"
langchain-huggingface = "^0.1.2"
transformers = "^4.46.2"
pandas = "^2.2.3"
plotly = "^5.24.1"
gradio = "^6.2.0"
```

## 🔍 Cas d'usage

- **Tutoring automatisé** : Répondre 24/7 aux questions des étudiants
- **Complément pédagogique** : Expliquer les concepts du cours
- **Support étudiant** : Fournir des clarifications rapides
- **Feedback personnalisé** : Adapter les réponses au contexte de la conversation

## 🐛 Dépannage

### Le modèle ne charge pas
- Vérifier la clé API OpenAI dans le fichier `.env`
- S'assurer que la clé a les bonnes permissions

### Pas de résultats de recherche
- Vérifier que les fichiers PDF sont dans le dossier `data/`
- Vérifier que la base de données Chroma a été initialisée
- Augmenter `CHUNK_OVERLAP` pour plus de flexibilité

### Problèmes de mémoire
- Réduire `CHUNK_SIZE` pour des segments plus petits
- Réduire le nombre de documents traités
- Augmenter l'allocation de mémoire RAM

## 📝 Licence

Voir le fichier [LICENSE](LICENSE) pour les détails.

## 👤 Auteurs

- **MBIA NDI Marie Thérèse** - Créatrice principale
  - Email: [mbialaura12@gmail.com](mailto:mbialaura12@gmail.com)

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le repository
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📞 Support

Pour toute question ou problème, veuillez :
- Ouvrir une issue sur GitHub
- Envoyer un email à [mbialaura12@gmail.com](mailto:mbialaura12@gmail.com)
