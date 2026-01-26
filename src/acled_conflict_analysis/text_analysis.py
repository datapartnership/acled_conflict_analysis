# Run this cell ONCE to download required NLTK data
import ssl
import certifi
import pandas as pd
import numpy as np
from wbpyplot import wb_plot

# Set SSL context to unverified BEFORE any NLTK imports
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Now import NLTK
import nltk

# Download required NLTK data
print("Downloading NLTK data (with SSL verification disabled)...")

resources = [
    'stopwords',
    'punkt',
    'punkt_tab',  # Newer NLTK versions require this
    'wordnet',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',  # Newer version
    'omw-1.4'
]

for resource in resources:
    try:
        print(f"\nDownloading {resource}...")
        nltk.download(resource, quiet=False)
        print(f"✓ {resource} downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading {resource}: {e}")

# Define custom stopwords for protest analysis
# These are common words in protest descriptions that don't add meaningful information

custom_stopwords = [
    # Generic protest terms
    'protest', 'protester', 'protesters', 'demonstration', 'demonstrator', 'demonstrators',
    'rally', 'march', 'marched', 'marching', 'gathered', 'gathering','cause', 'movement', 'movement','sit-in','sit ins','sit in',
    
    # Common report words
    'report', 'reported', 'reporting', 'according', 'source', 'sources',
    
    # Location/time words (if not informative)
    'area', 'location', 'place', 'time', 'day', 'week', 'month', 'year',
    'today', 'yesterday', 'recently', 'months', 'county',
    
    # Action words (common verbs in protest descriptions)
    'took', 'take', 'taken', 'taking',
    'made', 'make', 'making',
    'held', 'hold', 'holding',
    'called', 'call', 'calling',
    'said', 'say', 'saying',
    'stated', 'state', 'stating',
    'claimed', 'claim', 'claiming',
    'demanded', 'demand', 'demanding',
    'gathered', 'gather', 'gathering',
    'rallied', 'rally', 'rallying',
    'marched', 'march', 'marching',
    'protested', 'protest', 'protesting',
    'chanted', 'chant', 'chanting', 'chants',
    'staged', 'stage', 'staging',
    'organized', 'organize', 'organizing',
    'participated', 'participate', 'participating',
    'attended', 'attend', 'attending',
    'joined', 'join', 'joining',
    'blocked', 'block', 'blocking',
    'opposed', 'oppose', 'opposing',
    'supported', 'support', 'supporting',
    'condemned', 'condemn', 'condemning',
    'criticized', 'criticize', 'criticizing',
    'denounced', 'denounce', 'denouncing',
    'expressed', 'express', 'expressing',
    'voiced', 'voice', 'voicing',
    'demonstrated', 'demonstrate', 'demonstrating',
    'assembled', 'assemble', 'assembling',
    'convened', 'convene', 'convening',
    'coordinated', 'coordinate', 'coordinating',
    'followed', 'follow', 'following',
    'continued', 'continue', 'continuing', 'ongoing',
    'called', 'call', 'calling','case','protest',

    
    # Numbers and determiners (if not removed already)
    'one', 'two', 'three', 'several', 'many', 'number', 'group', 'groups',
    
    # ACLED-specific terms
    'event', 'events', 'acled', 'data','coded',
    
    # Common but uninformative words
    'people', 'person', 'individual', 'individuals',
    'also', 'however', 'although', 'though', 'great', 'front', 
    'district', 'tehran', 'iran', 'resident', 'residents', 
    'isfahan', 'city', 'local', 'khuzestan', 'near', 'outside',
    'office', 'province', 'building', 'organization', 'provincial',
    'regarding', 'central', 'institution', 'ground',
    'protesters', 'slogans', 'friday', 'anti', 'recent', 'lack',
    'activists', 'activity', 'activities', 'security', 'forces',
    'large', 'small', 'several', 'various', 'numerous','round',
    'part', 'members', 'member', 'team', 'unit', 'units',"context",'town', 'city','civil', 'society',

    # Month names
    'january', 'february', 'march', 'april', 'may', 'june', 'club', 
    'july', 'august', 'september', 'october', 'november', 'december','s', 'al', 'sit', 'latest', 'actions',
    'across','u','government','including','end',

    # Places
    'sindh', 'karachi', 'lahore', 'punjab',

    # Prepositions and common function words
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'by', 'about', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'around',
    'within', 'without', 'against', 'among', 'towards', 'upon', 'behind', 'beside',
    'throughout', 'amid', 'via', 'per', 'till', 'until', 'since',

    # Politically Charged words
    'zionist', 'zionism', 'morocco','moroccan', 'road', 'roads', 'street', 'streets', 'dozen', 'dozens',
    
    # Additional stopwords
    'national', 'start', 'started', 'starting', 'starts',
    
    # Add your own domain-specific stopwords here
    # Example: 'covid', 'pandemic' if not relevant to your analysis
]

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatizer = WordNetLemmatizer()



def generate_word_cloud_from_column(df, column_name, custom_stopwords):
    # Combine all the text in the specified column into a single string
    text = " ".join(note for note in df[column_name])

    stopwords = set(STOPWORDS)
    if custom_stopwords:
        stopwords.update(custom_stopwords)  

    # Generate the word cloud
     # Generate the word cloud, excluding the stopwords
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=stopwords  # Pass the stopwords set here
    ).generate(text)


    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")  # Hide the axes
    plt.show()


def get_word_counts(df, column_name, custom_stopwords):

    # Word normalization dictionary - maps variations to canonical form
    # This groups related words like morocco/moroccan, palestine/palestinian
    word_normalizations = {
        'moroccan': 'morocco',
        'moroccans': 'morocco',
        'palestinian': 'palestine',
        'palestinians': 'palestine',
        'israeli': 'israel',
        'israelis': 'israel',
        'egyptian': 'egypt',
        'egyptians': 'egypt',
        'iraqi': 'iraq',
        'iraqis': 'iraq',
        'syrian': 'syria',
        'syrians': 'syria',
        'lebanese': 'lebanon',
        'yemeni': 'yemen',
        'yemenis': 'yemen',
        'tunisian': 'tunisia',
        'tunisians': 'tunisia',
        'algerian': 'algeria',
        'algerians': 'algeria',
        'libyan': 'libya',
        'libyans': 'libya',
        'jordanian': 'jordan',
        'jordanians': 'jordan',
        'afghan': 'afghanistan',
        'afghans': 'afghanistan',
        'pakistani': 'pakistan',
        'pakistanis': 'pakistan',
        'saudi': 'saudi arabia',
        'saudis': 'saudi arabia',
        'emirati': 'uae',
        'emiratis': 'uae',
        # Add more as needed
        'salaries': 'salary',
        'retirees': 'retirement',
        'retired': 'retirement',
    }
    
    text = " ".join(note for note in df[column_name])
    words = re.findall(r'\b\w+\b', text.lower())

    wordcloud_stopwords = STOPWORDS
    all_stopwords = wordcloud_stopwords.union(custom_stopwords)
    all_stopwords_lower = {word.lower() for word in all_stopwords}

    # Filter out stopwords and numeric values
    filtered_words = [
        word for word in words 
        if word.lower() not in all_stopwords_lower and not word.isnumeric()
    ]
    
    # Use lemmatization instead of stemming to preserve word forms better
    # This keeps nouns like "workers" separate from verbs like "working"
    # Lemmatization respects part of speech, so it won't over-reduce words
    lemmatized_words = [lemmatizer.lemmatize(word, pos='n') for word in filtered_words]
    
    # Apply word normalizations to group related terms
    normalized_words = [word_normalizations.get(word, word) for word in lemmatized_words]

    # Count the occurrences of each normalized word
    word_counts = Counter(normalized_words)

    # Create DataFrame
    word_count_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
    word_count_df = word_count_df.sort_values(by='Count', ascending=False)

    return word_count_df


def get_ngram_counts(df, column_name, custom_stopwords, n=2, top_n=50):
    """
    Extract n-grams (phrases of n words) from text column.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column_name : str
        Name of text column
    custom_stopwords : list
        Custom stopwords to filter
    n : int
        Number of words in phrase (2=bigrams, 3=trigrams)
    top_n : int
        Number of top results to return
        
    Returns:
    --------
    DataFrame with ngram phrases and their counts
    """
    from nltk import word_tokenize
    
    all_ngrams = []
    
    # Process each row separately to maintain phrase boundaries
    for text in df[column_name].dropna():
        text = str(text).lower()
        
        # Tokenize properly
        words = word_tokenize(text)
        
        # Filter stopwords and non-alphabetic
        wordcloud_stopwords = STOPWORDS
        all_stopwords = wordcloud_stopwords.union(custom_stopwords)
        all_stopwords_lower = {word.lower() for word in all_stopwords}
        
        filtered_words = [
            word for word in words 
            if word.isalpha() and word not in all_stopwords_lower and len(word) > 2
        ]
        
        # Generate n-grams
        if len(filtered_words) >= n:
            text_ngrams = list(ngrams(filtered_words, n))
            all_ngrams.extend(text_ngrams)
    
    # Count n-grams
    ngram_counts = Counter(all_ngrams)
    
    # Create DataFrame
    ngram_df = pd.DataFrame(
        [(' '.join(ngram), count) for ngram, count in ngram_counts.most_common(top_n)],
        columns=['Phrase', 'Count']
    )
    
    return ngram_df


def get_collocations(df, column_name, custom_stopwords, n=2, top_n=50, min_freq=5):
    """
    Find collocations (words that frequently occur together) using statistical measures.
    This is more sophisticated than simple n-gram counting - it finds words that 
    appear together more often than would be expected by chance.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    column_name : str
        Name of text column
    custom_stopwords : list
        Custom stopwords to filter
    n : int
        2 for bigrams, 3 for trigrams
    top_n : int
        Number of top results to return
    min_freq : int
        Minimum frequency threshold
        
    Returns:
    --------
    DataFrame with collocations and their PMI scores
    """
    from nltk import word_tokenize
    
    # Combine all text
    all_words = []
    
    for text in df[column_name].dropna():
        text = str(text).lower()
        words = word_tokenize(text)
        
        # Filter
        wordcloud_stopwords = STOPWORDS
        all_stopwords = wordcloud_stopwords.union(custom_stopwords)
        all_stopwords_lower = {word.lower() for word in all_stopwords}
        
        filtered = [
            word for word in words 
            if word.isalpha() and word not in all_stopwords_lower and len(word) > 2
        ]
        all_words.extend(filtered)
    
    # Find collocations
    if n == 2:
        finder = BigramCollocationFinder.from_words(all_words)
        finder.apply_freq_filter(min_freq)
        # Use PMI (Pointwise Mutual Information) - measures how much more likely 
        # words are to occur together vs independently
        scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    else:  # n == 3
        finder = TrigramCollocationFinder.from_words(all_words)
        finder.apply_freq_filter(min_freq)
        scored = finder.score_ngrams(TrigramAssocMeasures.pmi)
    
    # Create DataFrame
    results = []
    for ngram, score in scored[:top_n]:
        phrase = ' '.join(ngram)
        # Get actual frequency
        freq = finder.ngram_fd[ngram]
        results.append((phrase, freq, round(score, 2)))
    
    colloc_df = pd.DataFrame(results, columns=['Phrase', 'Frequency', 'PMI_Score'])
    
    return colloc_df

def get_word_counts_improved(df, column_name, custom_stopwords):
    """
    Improved word count function with proper processing order:
    1. Tokenize
    2. Clean & lemmatize 
    3. Check stopwords (on cleaned forms)
    4. Normalize
    """
    
    # Word normalization dictionary - same as before
    word_normalizations = {
        'moroccan': 'morocco', 'moroccans': 'morocco',
        'palestinian': 'palestine', 'palestinians': 'palestine',
        'israeli': 'israel', 'israelis': 'israel',
        'egyptian': 'egypt', 'egyptians': 'egypt',
        'iraqi': 'iraq', 'iraqis': 'iraq',
        'syrian': 'syria', 'syrians': 'syria',
        'lebanese': 'lebanon',
        'yemeni': 'yemen', 'yemenis': 'yemen',
        'tunisian': 'tunisia', 'tunisians': 'tunisia',
        'algerian': 'algeria', 'algerians': 'algeria',
        'libyan': 'libya', 'libyans': 'libya',
        'jordanian': 'jordan', 'jordanians': 'jordan',
        'afghan': 'afghanistan', 'afghans': 'afghanistan',
        'pakistani': 'pakistan', 'pakistanis': 'pakistan',
        'saudi': 'saudi arabia', 'saudis': 'saudi arabia',
        'emirati': 'uae', 'emiratis': 'uae',
        'salaries': 'salary',
        'retirees': 'retirement', 'retired': 'retirement',
    }
    
    text = " ".join(note for note in df[column_name])
    words = re.findall(r'\b\w+\b', text.lower())

    # Prepare stopwords
    wordcloud_stopwords = STOPWORDS
    all_stopwords = wordcloud_stopwords.union(custom_stopwords)
    all_stopwords_lower = {word.lower() for word in all_stopwords}
    
    processed_words = []
    
    for word in words:
        # Skip numeric values
        if word.isnumeric():
            continue
            
        # Lemmatize first (this converts "dozens" → "dozen")
        lemmatized = lemmatizer.lemmatize(word, pos='n')
        
        # Then check stopwords on the lemmatized form
        if lemmatized.lower() in all_stopwords_lower:
            continue
            
        # Apply normalizations
        normalized = word_normalizations.get(lemmatized, lemmatized)
        processed_words.append(normalized)

    # Count the occurrences
    word_counts = Counter(processed_words)

    # Create DataFrame
    word_count_df = pd.DataFrame(word_counts.items(), columns=['Word', 'Count'])
    word_count_df = word_count_df.sort_values(by='Count', ascending=False)

    return word_count_df

print("✓ Improved word count function created - lemmatizes BEFORE stopword filtering")


def plot_wordcloud_comparison(
    word_freq_dicts,
    subplot_titles=None,
    main_title="Word Cloud Comparison",
    subtitle="",
    note="",
    figsize=(12, 6),
    ncols=2,
    wordcloud_width=400,
    wordcloud_height=300,
    background_color='white',
    max_words=200
):
    """
    Create side-by-side word clouds for comparison using wb_plot styling.
    
    Parameters:
    -----------
    word_freq_dicts : list of dict
        List of dictionaries where each dict has {word: frequency} pairs.
        Example: [{'word1': 100, 'word2': 50}, {'word3': 80, 'word4': 60}]
    subplot_titles : list of str, optional
        List of titles for each subplot. Length should match word_freq_dicts.
        Example: ['2015-2020', '2021-2025']
    main_title : str, optional
        Main title for the entire figure. Default: "Word Cloud Comparison"
    subtitle : str, optional
        Subtitle for the figure. Default: ""
    note : str, optional
        Source note for the figure. Default: ""
    figsize : tuple, optional
        Figure size as (width, height). Default: (12, 6)
    ncols : int, optional
        Number of columns for subplots. Default: 2
    wordcloud_width : int, optional
        Width of each word cloud in pixels. Default: 400
    wordcloud_height : int, optional
        Height of each word cloud in pixels. Default: 300
    background_color : str, optional
        Background color for word clouds. Default: 'white'
    max_words : int, optional
        Maximum number of words to display in each word cloud. Default: 200
    
    Returns:
    --------
    None (displays plot using wb_plot styling)
    
    Example:
    --------
    >>> word_freq_1 = dict(zip(df1['Word'][:20], df1['Count'][:20]))
    >>> word_freq_2 = dict(zip(df2['Word'][:20], df2['Count'][:20]))
    >>> plot_wordcloud_comparison(
    ...     [word_freq_1, word_freq_2],
    ...     subplot_titles=['Period 1', 'Period 2'],
    ...     main_title='Word Frequency Comparison',
    ...     subtitle='Comparing word frequencies across time periods',
    ...     note='ACLED protest descriptions'
    ... )
    """
    from wordcloud import WordCloud
    
    n_plots = len(word_freq_dicts)
    nrows = int(np.ceil(n_plots / ncols))
    
    # Use wb_plot decorator for consistent styling
    @wb_plot(
        title=main_title,
        subtitle=subtitle,
        note=[("Source:", note)] if note else None,
    )
    def _plot(axs):
        # Get the figure and clear it to create our own layout
        if isinstance(axs, (list, tuple, np.ndarray)) and len(axs) > 0:
            fig = axs[0].figure
        else:
            fig = plt.gcf()
        fig.clf()
        fig.set_size_inches(figsize[0], figsize[1])
        
        # Create subplots
        axes = fig.subplots(nrows, ncols).flatten() if n_plots > 1 else [fig.subplots(1, 1)]
        
        # Generate word clouds for each dictionary
        for idx, word_freq in enumerate(word_freq_dicts):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=wordcloud_width,
                height=wordcloud_height,
                background_color=background_color,
                colormap='viridis',
                max_words=max_words
            ).generate_from_frequencies(word_freq)
            
            # Display word cloud
            ax.imshow(wordcloud, interpolation='bilinear')
            
            # Remove axes
            ax.axis('off')
            
            # Add subplot title if provided
            if subplot_titles and idx < len(subplot_titles):
                ax.set_title(subplot_titles[idx], fontsize=14, fontweight='bold', pad=10)
        
        # Hide any unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        # Adjust layout with gaps between plots
        plt.tight_layout(w_pad=3.0, h_pad=2.0)
    
    _plot()

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.stem import WordNetLemmatizer
from wordcloud import STOPWORDS

lemmatizer = WordNetLemmatizer()

def get_tfidf_word_frequencies(df, text_column, custom_stopwords, word_normalizations, top_n=100):
    """
    Extract top N words by TF-IDF score from a dataframe.
    
    Returns:
        dict: {word: tfidf_score} for top N words
    """
    # Prepare stopwords
    all_stopwords = set(STOPWORDS).union(set(custom_stopwords))
    all_stopwords_lower = {word.lower() for word in all_stopwords}
    
    def preprocess_text(text):
        """Preprocess text similar to topic modeling pipeline"""
        text = str(text).lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Filter stopwords and non-alphabetic
        filtered = [w for w in words if w not in all_stopwords_lower and w.isalpha() and len(w) > 2]
        
        # Lemmatize
        lemmatized = [lemmatizer.lemmatize(w, pos='n') for w in filtered]
        
        # Apply normalizations
        normalized = [word_normalizations.get(w, w) for w in lemmatized]
        
        return ' '.join(normalized)
    
    # Preprocess all texts
    processed_texts = df[text_column].apply(preprocess_text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=top_n * 3,  # Get more features than needed
        min_df=5,  # Ignore terms that appear in fewer than 5 documents
        max_df=0.7  # Ignore terms that appear in more than 70% of documents
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Get feature names and their average TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    
    # Create word-score dictionary
    word_scores = dict(zip(feature_names, mean_tfidf))
    
    # Sort by score and get top N
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return dict(sorted_words)

print("✓ TF-IDF word frequency extraction function created")