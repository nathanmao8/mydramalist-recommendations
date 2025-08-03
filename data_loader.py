import os
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Set
import time
import random

class DataLoader:
    """
    DataLoader handles all drama data loading and caching operations.
    
    Features:
    - Loads popular dramas from HTML files and caches them
    - Maintains separate cache for non-popular watched dramas  
    - Handles API rate limiting with automatic retries
    - Processes drama metadata, cast, crew, and reviews
    - Manages user watchlist data and ratings
    
    Cache Strategy:
    - Popular dramas: Single large cache file for 1000+ popular dramas
    - Non-popular watched: Dynamic cache that grows with user's watched list
    - Automatic retry mechanism for failed API calls
    """
    # File paths and configuration constants
    HTML_FOLDER = 'html_popular'
    POPULAR_CACHE_FILE = 'popular_dramas_processed.json'
    NON_POPULAR_CACHE_FILE = 'watched_non_popular_cache.json'
    FAILED_DRAMAS_FILE = 'failed_dramas.json'
    
    # Processing configuration
    HTML_FILE_COUNT = 75  # Number of popular*.html files (1-75)
    API_DELAY = 0.5  # Delay between API calls in seconds
    MAX_RETRIES = 3  # Maximum retry attempts for failed dramas
    RETRY_DELAY_BASE = 1  # Base delay for exponential backoff
    RETRY_DELAY_MAX = 3  # Maximum delay for exponential backoff
    RETRY_ROUND_DELAY = 30  # Delay between retry rounds in seconds
    
    def __init__(self, api):
        """
        Initialize the DataLoader with API instance.

        Parameters
        ----------
        api : object
            API client instance for making requests to MyDramaList API
            
        Returns
        -------
        None
        """
        self.api = api
        self.cache_file = self.POPULAR_CACHE_FILE
        self.non_popular_cache_file = self.NON_POPULAR_CACHE_FILE
        self.failed_dramas_file = self.FAILED_DRAMAS_FILE

    def get_cache_status(self) -> Dict[str, any]:
        """
        Get current status of all caches for debugging and monitoring.
        
        Returns
        -------
        Dict[str, any]
            Dictionary containing cache status information including:
            - popular_cache_exists: bool, whether popular cache file exists
            - non_popular_cache_exists: bool, whether non-popular cache exists
            - failed_dramas_file_exists: bool, whether failed dramas file exists
            - popular_cache_size: int or str, number of dramas in popular cache
            - non_popular_cache_size: int or str, number of dramas in non-popular cache
            - failed_dramas_count: int or str, number of failed dramas
        """
        status = {
            'popular_cache_exists': os.path.exists(self.cache_file),
            'non_popular_cache_exists': os.path.exists(self.non_popular_cache_file),
            'failed_dramas_file_exists': os.path.exists(self.failed_dramas_file),
            'popular_cache_size': 0,
            'non_popular_cache_size': 0,
            'failed_dramas_count': 0
        }
        
        # Get cache sizes
        if status['popular_cache_exists']:
            try:
                popular_data = self.load_popular_dramas_cache()
                status['popular_cache_size'] = len(popular_data)
            except:
                status['popular_cache_size'] = 'error_loading'
        
        if status['non_popular_cache_exists']:
            try:
                non_popular_data = self.load_non_popular_watched_cache()
                status['non_popular_cache_size'] = len(non_popular_data)
            except:
                status['non_popular_cache_size'] = 'error_loading'
        
        if status['failed_dramas_file_exists']:
            try:
                with open(self.failed_dramas_file, 'r') as f:
                    failed_data = json.load(f)
                    status['failed_dramas_count'] = failed_data.get('total_failed', 0)
            except:
                status['failed_dramas_count'] = 'error_loading'
        
        return status

    def get_popular_drama_slugs(self) -> Set[str]:
        """
        Extract drama slugs from all popular*.html files in the specified folder.
        
        Files contain the 1000 most popular dramas and 500 most popular movies 
        based on MyDramaList.com's rankings on July 17, 2025.
        
        Returns
        -------
        Set[str]
            Set of all drama slugs found across all HTML files
            
        Notes
        -----
        Processes files named popular1.html through popular75.html in the 
        HTML_FOLDER directory. Each file contains drama links with slugs 
        extracted from href attributes.
        """
        folder_path = self.HTML_FOLDER
        all_slugs = set()
        
        # Loop through all 75 files
        for i in range(1, self.HTML_FILE_COUNT + 1):
            filename = f'popular{i}.html'
            filepath = os.path.join(folder_path, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    
                    # Find all h6 elements with class "text-primary title"
                    title_elements = soup.find_all('h6', class_='text-primary title')
                    
                    for title_element in title_elements:
                        # Find the anchor tag within this title element
                        link = title_element.find('a', href=True)
                        if link:
                            href = link['href']
                            # Remove the leading '/' to get just the slug
                            slug = href[1:]  # Remove the first character '/'
                            all_slugs.add(slug)
                            
            except FileNotFoundError:
                print(f"Warning: {filename} not found")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        return all_slugs
    
    def load_user_watchlist(self, user_id: str) -> Tuple[Dict, Set[str]]:
        """
        Load user's rated dramas from watchlist and return watched slugs.
        
        Parameters
        ----------
        user_id : str
            MyDramaList user identifier
            
        Returns
        -------
        Tuple[Dict, Set[str]]
            Tuple containing:
            - watchlist_data: Dict, complete watchlist data from API (None if user doesn't exist)
            - watched_slugs: Set[str], set of drama slugs from completed items
            
        Notes
        -----
        Only processes items from the 'Completed' section of the user's 
        drama list. Returns None for watchlist_data if user doesn't exist,
        empty dict and set if API call fails for other reasons.
        """
        try:
            watchlist_data = self.api.get_user_dramalist(user_id)
            watched_slugs = set()
            
            # If API returns None, it means the user doesn't exist (404 error)
            if watchlist_data is None:
                return None, set()
            
            if watchlist_data and watchlist_data.get('data'):
                # Access the 'list' within data, then 'Completed' items
                drama_list = watchlist_data.get('data', {}).get('list', {})
                completed_items = drama_list.get('Completed', {}).get('items', [])
                
                for item in completed_items:
                    slug = item.get('id', '')
                    if slug:
                        watched_slugs.add(slug)
            
            return watchlist_data, watched_slugs
        except Exception as e:
            print(f"Error loading user watchlist for {user_id}: {e}")
            return {}, set()
    
    def load_drama_details(self, slug: str) -> Dict:
        """
        Load comprehensive drama information from multiple API endpoints.
        
        Parameters
        ----------
        slug : str
            Drama slug identifier for API requests
            
        Returns
        -------
        Dict
            Dictionary containing:
            - info: Dict, basic drama information
            - cast: Dict, cast and crew information  
            - reviews: Dict, user reviews and ratings
            
        Raises
        ------
        Exception
            If any API call fails, the exception is re-raised after logging
        """
        try:
            drama_info = self.api.get_drama_info(slug)
            cast_info = self.api.get_cast(slug)
            reviews_info = self.api.get_reviews(slug)
            
            return {
                'info': drama_info,
                'cast': cast_info,
                'reviews': reviews_info
            }
        except Exception as e:
            print(f"Error loading drama details for {slug}: {e}")
            raise
    
    def _extract_categorical_features(self, drama_details: Dict) -> Tuple[str, str, str, List, List, str, str]:
        """
        Extract categorical/text-based information from drama details.
        
        Parameters
        ----------
        drama_details : Dict
            Complete drama details containing info, cast, and reviews
            
        Returns
        -------
        Tuple[str, str, str, List, List, str, str]
            Tuple containing:
            - title: str, drama title
            - slug: str, drama slug identifier
            - synopsis: str, cleaned synopsis text
            - genres: List[str], list of genre categories
            - tags: List[str], list of drama tags
            - country: str, country of origin
            - drama_type: str, type of drama (TV, movie, etc.)
            
        Notes
        -----
        Synopsis is cleaned by removing source attribution text that appears
        after "(Source:" patterns. Tags list has extra text removed from
        the last element.
        """
        drama_info = drama_details.get('info', {}).get('data', {})
        
        # Basic info
        title = drama_info.get('title', '')
        slug = drama_details.get('info', {}).get('slug_query', '')
        
        # Extract and clean synopsis, remove "(Source: )" text
        raw_synopsis = drama_info.get('synopsis', '')
        splits = raw_synopsis.split('\n(')
        synopsis = splits[0]
        
        # Metadata
        others_section = drama_info.get('others', {})
        genres = others_section.get('genres', [])
        raw_tags = others_section.get('tags', [])
        
        # Handle extra text at end of last tag
        if raw_tags:
            last = raw_tags[-1:][0][:-12]
            tags = raw_tags[:-1]
            tags.append(last)
        else:
            tags = []
        
        # Location info
        details_section = drama_info.get('details', {})
        country = details_section.get('country', '')
        drama_type = details_section.get('type', '')
        
        return title, slug, synopsis, genres, tags, country, drama_type
    
    def _extract_numerical_features(self, drama_details: Dict) -> Tuple[float, float, float]:
        """
        Extract and convert numerical features with proper error handling.
        
        Parameters
        ----------
        drama_details : Dict
            Complete drama details containing info, cast, and reviews
            
        Returns
        -------
        Tuple[float, float, float]
            Tuple containing:
            - year_num: float, release year as number
            - rating_num: float, drama rating score
            - watchers_num: float, number of watchers/viewers
            
        Notes
        -----
        All values are safely converted to float with fallback to 0.0 
        for invalid or missing data. Watchers field has commas removed
        before conversion.
        """
        drama_info = drama_details.get('info', {}).get('data', {})
        details_section = drama_info.get('details', {})
        
        year = drama_info.get('year', '')
        drama_rating = drama_info.get('rating', '')
        watchers = details_section.get('watchers', '')
        
        # Convert numerical features to float, with fallback to 0
        year_num = self._safe_float_conversion(year)
        rating_num = self._safe_float_conversion(drama_rating)
        
        # Handle watchers with comma removal
        try:
            watchers_clean = str(watchers).replace(',', '') if watchers else '0'
            watchers_num = float(watchers_clean)
        except (ValueError, TypeError):
            watchers_num = 0.0
            
        return year_num, rating_num, watchers_num
    
    def _safe_float_conversion(self, value) -> float:
        """
        Safely convert a value to float with fallback to 0.0.
        
        Parameters
        ----------
        value : Any
            Value to convert to float (string, number, or other type)
            
        Returns
        -------
        float
            Converted float value, or 0.0 if conversion fails
            
        Notes
        -----
        Handles None values, empty strings, and invalid number formats
        gracefully by returning 0.0.
        """
        try:
            return float(value) if value else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def extract_and_clean_drama_features(self, drama_details: Dict, text_processor) -> Dict:
        """
        Extract and clean all necessary features from drama details.
        
        Parameters
        ----------
        drama_details : Dict
            Complete drama details containing info, cast, and reviews
        text_processor : object
            TextProcessor instance for text cleaning and sentiment analysis
            
        Returns
        -------
        Dict or None
            Dictionary containing all extracted features:
            - slug: str, drama identifier
            - title: str, drama title
            - synopsis_clean: str, cleaned synopsis text
            - reviews_combined: str, processed review text
            - review_helpfulness_weights: List[float], review helpfulness scores
            - genres: List[str], genre categories
            - tags: List[str], drama tags
            - main_cast: List[str], main cast member names
            - directors: List[str], director names
            - screenwriters: List[str], screenwriter names
            - composers: List[str], composer names
            - sentiment_features: Dict, sentiment analysis results
            - year: float, release year
            - drama_rating: float, rating score
            - country: str, country of origin
            - drama_type: str, drama type
            - watchers: float, number of watchers
            Returns None if processing fails.
            
        Notes
        -----
        Combines synopsis and reviews for sentiment analysis. Extracts cast
        and crew information from cast data. All text is processed through
        the provided text_processor for cleaning and feature extraction.
        """
        try:
            # Extract categorical information
            title, slug, synopsis, genres, tags, country, drama_type = self._extract_categorical_features(drama_details)
            synopsis_clean = text_processor.clean_text(synopsis)
            
            # Extract and process reviews
            reviews_combined, helpfulness_weights = text_processor.process_reviews(
                drama_details.get('reviews', {})
            )
            
            # Extract cast and crew
            main_cast = self.extract_main_cast(drama_details.get('cast', {}))
            crew_data = self._extract_crew(drama_details.get('cast', {}))
            
            # Extract numerical features
            year_num, rating_num, watchers_num = self._extract_numerical_features(drama_details)
            
            # Extract sentiment features
            combined_text = synopsis_clean + ' ' + reviews_combined
            sentiment_features = text_processor.extract_sentiment_features(combined_text)
            
            return {
                'slug': slug,
                'title': title,
                'synopsis_clean': synopsis_clean,
                'reviews_combined': reviews_combined,
                'review_helpfulness_weights': helpfulness_weights,
                'genres': genres,
                'tags': tags,
                'main_cast': main_cast,
                'directors': crew_data.get('directors', []),
                'screenwriters': crew_data.get('screenwriters', []),
                'composers': crew_data.get('composers', []),
                'sentiment_features': sentiment_features,
                'year': year_num,
                'drama_rating': rating_num,
                'country': country,
                'drama_type': drama_type,
                'watchers': watchers_num
            }
        except Exception as e:
            print(f"Error processing drama {drama_details.get('info', {}).get('slug_query', 'unknown')}: {e}")
            return None
    
    def extract_main_cast(self, cast_data: Dict) -> List[str]:
        """
        Extract main role cast members with error handling.
        
        Parameters
        ----------
        cast_data : Dict
            Cast data dictionary containing cast and crew information
            
        Returns
        -------
        List[str]
            List of main cast member names, empty list if no data or error
            
        Notes
        -----
        Only extracts actors from the 'Main Role' category. Other cast
        categories like 'Support Role' are ignored.
        """
        try:
            if not cast_data or not cast_data.get('data'):
                return []
            
            main_cast = []
            # Access the 'casts' section within 'data'
            casts_section = cast_data.get('data', {}).get('casts', {})
            main_role_actors = casts_section.get('Main Role', [])
            
            for actor in main_role_actors:
                actor_name = actor.get('name', '')
                if actor_name:
                    main_cast.append(actor_name)
            
            return main_cast
        except Exception as e:
            print(f"Error extracting main cast: {e}")
            return []
    
    def _process_single_drama(self, slug: str, text_processor, drama_number: int = None, total_dramas: int = None) -> Dict:
        """
        Process a single drama and return the processed data or None if failed.
        
        Parameters
        ----------
        slug : str
            Drama slug identifier to process
        text_processor : object
            TextProcessor instance for text cleaning and analysis
        drama_number : int, optional
            Current drama number for progress tracking
        total_dramas : int, optional
            Total number of dramas being processed
            
        Returns
        -------
        Dict or None
            Processed drama data dictionary, or None if processing failed
            
        Notes
        -----
        Includes API delay to avoid overwhelming the server. Progress 
        information is printed if drama_number and total_dramas are provided.
        """
        try:
            if drama_number and total_dramas:
                print(f"Processing drama {drama_number}/{total_dramas}: {slug}")
            else:
                print(f"Processing drama: {slug}")
            
            # Add delay to avoid overwhelming the API
            time.sleep(self.API_DELAY)
            
            drama_details = self.load_drama_details(slug)
            processed_drama = self.extract_and_clean_drama_features(drama_details, text_processor)
            
            return processed_drama
        except Exception as e:
            print(f"Failed to process {slug}: {e}")
            return None

    def _process_drama_batch(self, slugs: List[str], text_processor) -> Tuple[Dict, List]:
        """
        Process a batch of drama slugs and return processed dramas and failed slugs.
        
        Parameters
        ----------
        slugs : List[str]
            List of drama slugs to process
        text_processor : object
            TextProcessor instance for text cleaning and analysis
            
        Returns
        -------
        Tuple[Dict, List]
            Tuple containing:
            - processed_dramas: Dict[str, Dict], successfully processed dramas
            - failed_dramas: List[str], slugs that failed to process
            
        Notes
        -----
        Processes dramas sequentially with progress tracking. Failed dramas
        are collected for potential retry operations.
        """
        processed_dramas = {}
        failed_dramas = []
        
        for i, slug in enumerate(slugs, 1):
            processed_drama = self._process_single_drama(slug, text_processor, i, len(slugs))
            
            if processed_drama:
                processed_dramas[slug] = processed_drama
            else:
                failed_dramas.append(slug)
        
        return processed_dramas, failed_dramas

    def _retry_failed_dramas(self, failed_dramas: List[str], text_processor, 
                           max_retries: int = None) -> Tuple[Dict, List]:
        """
        Handle retry logic for failed dramas with exponential backoff.
        
        Parameters
        ----------
        failed_dramas : List[str]
            List of drama slugs that failed initial processing
        text_processor : object
            TextProcessor instance for text cleaning and analysis
        max_retries : int, optional
            Maximum number of retry attempts, defaults to class MAX_RETRIES
            
        Returns
        -------
        Tuple[Dict, List]
            Tuple containing:
            - processed_dramas: Dict[str, Dict], successfully processed dramas
            - remaining_failures: List[str], slugs that still failed after retries
            
        Notes
        -----
        Uses exponential backoff with randomization between retry attempts.
        Includes delays between retry rounds to avoid overwhelming the API.
        Progress is reported for each retry attempt.
        """
        if max_retries is None:
            max_retries = self.MAX_RETRIES
            
        print(f"\nStarting automatic retry for {len(failed_dramas)} failed dramas...")
        
        processed_dramas = {}
        remaining_failures = failed_dramas.copy()
        retry_count = 0
        
        while remaining_failures and retry_count < max_retries:
            retry_count += 1
            print(f"Retry attempt {retry_count}/{max_retries}")
            
            current_failures = remaining_failures.copy()
            remaining_failures.clear()
            
            for i, slug in enumerate(current_failures, 1):
                try:
                    print(f"Retrying drama {i}/{len(current_failures)}: {slug}")
                    
                    # Exponential backoff with randomization
                    delay = random.uniform(self.RETRY_DELAY_BASE, self.RETRY_DELAY_MAX) * retry_count
                    time.sleep(delay)
                    
                    drama_details = self.load_drama_details(slug)
                    processed_drama = self.extract_and_clean_drama_features(drama_details, text_processor)
                    
                    if processed_drama:
                        processed_dramas[slug] = processed_drama
                        print(f"Successfully processed {slug} on retry {retry_count}")
                    else:
                        remaining_failures.append(slug)
                        
                except Exception as e:
                    print(f"Retry failed for {slug}: {e}")
                    remaining_failures.append(slug)
            
            if remaining_failures:
                print(f"Still have {len(remaining_failures)} failed dramas after retry {retry_count}")
                if retry_count < max_retries:
                    print(f"Waiting {self.RETRY_ROUND_DELAY} seconds before next retry...")
                    time.sleep(self.RETRY_ROUND_DELAY)
        
        return processed_dramas, remaining_failures

    def create_popular_dramas_cache(self, text_processor, retry_on_error=True, max_retries=3) -> Dict:
        """
        Create cache with built-in automatic retry mechanism.
        
        Parameters
        ----------
        text_processor : object
            TextProcessor instance for text cleaning and analysis
        retry_on_error : bool, default True
            Whether to automatically retry failed dramas
        max_retries : int, default 3
            Maximum number of retry attempts for failed dramas
            
        Returns
        -------
        Dict
            Dictionary of successfully processed dramas with slug as key
            
        Notes
        -----
        Processes all popular dramas found in HTML files. Failed dramas
        are automatically retried if retry_on_error is True. Results are
        saved to cache files and comprehensive reports are generated.
        """
        
        popular_slugs = self.get_popular_drama_slugs()
        
        # First pass: process all dramas
        processed_dramas, failed_dramas = self._process_drama_batch(list(popular_slugs), text_processor)
        
        # Automatic retry mechanism for failed dramas
        if retry_on_error and failed_dramas:
            retry_processed, final_failures = self._retry_failed_dramas(failed_dramas, text_processor, max_retries)
            
            # Merge retry results with original results
            processed_dramas.update(retry_processed)
            failed_dramas = final_failures
        
        # Save results
        self._save_cache_results(processed_dramas, failed_dramas)
        
        # Report final results
        self._report_final_results(processed_dramas, failed_dramas, len(popular_slugs))
        
        return processed_dramas
    
    def _save_cache_results(self, processed_dramas: Dict, failed_dramas: List):
        """
        Save processed dramas and failed dramas list to files.
        
        Parameters
        ----------
        processed_dramas : Dict
            Dictionary of successfully processed dramas
        failed_dramas : List[str]
            List of drama slugs that failed processing
            
        Returns
        -------
        None
            
        Notes
        -----
        Saves successful dramas to the main cache file and failed dramas
        to a separate file for manual review. Failed dramas file includes
        timestamp and total count.
        """
        
        # Save successful dramas
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(processed_dramas, f, indent=2, ensure_ascii=False)
        
        # Save failed dramas list for manual review
        if failed_dramas:
            with open(self.failed_dramas_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'failed_slugs': failed_dramas,
                    'total_failed': len(failed_dramas),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
    
    def _report_final_results(self, processed_dramas: Dict, failed_dramas: List, total_dramas: int):
        """
        Print comprehensive results report for cache creation.
        
        Parameters
        ----------
        processed_dramas : Dict
            Dictionary of successfully processed dramas
        failed_dramas : List[str]
            List of drama slugs that failed processing
        total_dramas : int
            Total number of dramas that were attempted
            
        Returns
        -------
        None
            
        Notes
        -----
        Prints detailed statistics including success rate, failure count,
        and lists failed dramas for debugging. Shows first 10 failed dramas
        if there are more than 10 failures.
        """
        
        success_count = len(processed_dramas)
        failure_count = len(failed_dramas)
        success_rate = (success_count / total_dramas) * 100
        
        print("\n" + "="*60)
        print("CACHE CREATION RESULTS")
        print("="*60)
        print(f"Successfully processed: {success_count}/{total_dramas} ({success_rate:.1f}%)")
        print(f"Failed to process: {failure_count}/{total_dramas} ({(failure_count/total_dramas)*100:.1f}%)")
        
        if failed_dramas:
            print(f"\nFailed dramas saved to: {self.failed_dramas_file}")
            print("Failed dramas:")
            for slug in failed_dramas[:10]:  # Show first 10
                print(f"  - {slug}")
            if len(failed_dramas) > 10:
                print(f"  ... and {len(failed_dramas) - 10} more")
        
        print(f"💾 Cache saved to: {self.cache_file}")

    def _get_text_processor(self):
        """
        Import and return TextProcessor instance to avoid circular imports.
        
        Returns
        -------
        TextProcessor
            New instance of TextProcessor class
            
        Notes
        -----
        Uses late import to avoid circular dependency issues between
        DataLoader and TextProcessor classes.
        """
        from text_processor import TextProcessor
        return TextProcessor()

    def load_popular_dramas_cache(self) -> Dict:
        """
        Load popular dramas from cache file with improved error handling.
        
        Returns
        -------
        Dict
            Dictionary of cached drama data with slug as key, 
            empty dict if file doesn't exist or loading fails
            
        Notes
        -----
        Handles JSON decode errors and file access issues gracefully.
        Prints status messages for debugging and monitoring.
        """
        try:
            if not os.path.exists(self.cache_file):
                print(f"Cache file {self.cache_file} not found")
                return {}
                
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                print(f"Successfully loaded {len(cache_data)} dramas from cache")
                return cache_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from cache file: {e}")
            return {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}

    def load_non_popular_watched_cache(self) -> Dict:
        """
        Load the cache of non-popular watched dramas with improved error handling.
        
        Returns
        -------
        Dict
            Dictionary of cached non-popular drama data with slug as key,
            empty dict if file doesn't exist or loading fails
            
        Notes
        -----
        This cache contains dramas from user watchlists that are not in
        the popular dramas cache. Handles JSON decode errors and file
        access issues gracefully.
        """
        try:
            if not os.path.exists(self.non_popular_cache_file):
                print(f"Non-popular cache file {self.non_popular_cache_file} not found")
                return {}
                
            with open(self.non_popular_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                print(f"Successfully loaded {len(cache_data)} non-popular dramas from cache")
                return cache_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from non-popular cache file: {e}")
            return {}
        except Exception as e:
            print(f"Error loading non-popular cache: {e}")
            return {}

    def save_non_popular_watched_cache(self, cache_data: Dict):
        """
        Save the updated non-popular watched cache with error handling.
        
        Parameters
        ----------
        cache_data : Dict
            Dictionary of non-popular drama data to save
            
        Returns
        -------
        None
            
        Notes
        -----
        Saves data as JSON with UTF-8 encoding and pretty formatting.
        Prints success/error messages for debugging.
        """
        try:
            with open(self.non_popular_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(cache_data)} dramas to non-popular cache")
        except Exception as e:
            print(f"Error saving non-popular cache: {e}")

    def _ensure_popular_cache_exists(self) -> Dict:
        """
        Ensure popular cache exists, create if missing.
        
        Returns
        -------
        Dict
            Dictionary of popular drama data loaded from cache or newly created
            
        Notes
        -----
        Checks if popular cache file exists and loads it if available.
        If missing, automatically creates the cache by processing all
        popular drama HTML files.
        """
        if os.path.exists(self.cache_file):
            print(f"Loading popular dramas from cache: {self.cache_file}")
            return self.load_popular_dramas_cache()
        else:
            print("Cache file not found. Creating cache...")
            text_processor = self._get_text_processor()
            return self.create_popular_dramas_cache(text_processor)

    def _get_watched_drama_data(self, slug: str, popular_dramas: Dict, non_popular_cache: Dict) -> Dict:
        """
        Get drama data from either popular or non-popular cache, or fetch if not found.
        
        Parameters
        ----------
        slug : str
            Drama slug identifier to find
        popular_dramas : Dict
            Dictionary of popular drama data
        non_popular_cache : Dict
            Dictionary of non-popular drama data cache
            
        Returns
        -------
        Dict or None
            Drama data dictionary if found or successfully fetched,
            None if fetching fails
            
        Notes
        -----
        Searches popular cache first, then non-popular cache. If not found
        in either cache, fetches from API and adds to non-popular cache.
        Returns a copy of the data to prevent modification of cached data.
        """
        # Check popular cache first
        if slug in popular_dramas:
            return popular_dramas[slug].copy()
        
        # Check non-popular cache
        if slug in non_popular_cache:
            return non_popular_cache[slug].copy()
        
        # If not in either cache, fetch from API
        print(f"Fetching new non-popular watched drama: {slug}")
        try:
            drama_details = self.load_drama_details(slug)
            text_processor = self._get_text_processor()
            processed_drama = self.extract_and_clean_drama_features(drama_details, text_processor)
            
            # Add to non-popular cache
            if processed_drama:
                non_popular_cache[slug] = processed_drama
                return processed_drama.copy()
        except Exception as e:
            print(f"Failed to fetch details for new drama {slug}: {e}")
        
        return None

    def _process_watched_dramas(self, watchlist_data: Dict, popular_dramas: Dict, 
                               non_popular_cache: Dict) -> Tuple[List[Dict], Dict]:
        """
        Process watched dramas and add user ratings from watchlist.
        
        Parameters
        ----------
        watchlist_data : Dict
            Complete user watchlist data from API
        popular_dramas : Dict
            Dictionary of popular drama data
        non_popular_cache : Dict
            Dictionary of non-popular drama data cache
            
        Returns
        -------
        Tuple[List[Dict], Dict]
            Tuple containing:
            - watched_dramas: List[Dict], list of watched dramas with user ratings
            - non_popular_cache: Dict, updated non-popular cache
            
        Notes
        -----
        Only processes completed items from the watchlist. Validates user
        ratings and skips items with invalid scores. Adds user_rating field
        to each drama data dictionary.
        """
        watched_dramas = []
        completed_items = watchlist_data.get('data', {}).get('list', {}).get('Completed', {}).get('items', [])
        
        for item in completed_items:
            slug = item.get('id', '')
            score = item.get('score', '0.0')
            
            # Validate rating
            try:
                rating = float(score)
            except (ValueError, TypeError):
                print(f"Invalid rating for {slug}: {score}")
                continue
            
            # Get drama data
            processed_drama = self._get_watched_drama_data(slug, popular_dramas, non_popular_cache)
            
            # Add rating and append to watched list
            if processed_drama:
                processed_drama['user_rating'] = rating
                watched_dramas.append(processed_drama)
        
        return watched_dramas, non_popular_cache

    def _separate_unwatched_dramas(self, popular_dramas: Dict, watched_slugs: Set[str]) -> List[Dict]:
        """
        Extract unwatched dramas from popular dramas collection.
        
        Parameters
        ----------
        popular_dramas : Dict
            Dictionary of all popular drama data
        watched_slugs : Set[str]
            Set of drama slugs that the user has watched
            
        Returns
        -------
        List[Dict]
            List of unwatched drama data dictionaries
            
        Notes
        -----
        Filters popular dramas to exclude those already watched by the user.
        Used to create the recommendation pool for the user.
        """
        unwatched_dramas = []
        for slug, drama_data in popular_dramas.items():
            if slug not in watched_slugs:
                unwatched_dramas.append(drama_data)
        return unwatched_dramas

    def load_all_drama_data(self, user_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load all watched and unwatched drama data with a dual-caching system.
        
        Parameters
        ----------
        user_id : str
            MyDramaList user identifier
            
        Returns
        -------
        Tuple[List[Dict], List[Dict]]
            Tuple containing:
            - watched_dramas: List[Dict], dramas watched by user with ratings
            - unwatched_dramas: List[Dict], popular dramas not watched by user
            
        Notes
        -----
        Uses dual-caching strategy:
        - One cache for the 1000+ popular dramas from HTML files
        - Second cache for non-popular dramas from user's watchlist
        
        Automatically creates popular cache if missing. Non-popular cache
        grows dynamically as new dramas are encountered in user watchlists.
        """
        
        # Load user's watchlist and get watched slugs
        watchlist_data, watched_slugs = self.load_user_watchlist(user_id)
        print(f"User has watched {len(watched_slugs)} dramas")
        
        # Load or create popular dramas cache
        popular_dramas = self._ensure_popular_cache_exists()
        
        # Load non-popular watched cache
        non_popular_watched_cache = self.load_non_popular_watched_cache()
        
        # Process watched dramas with ratings
        watched_dramas, updated_non_popular_cache = self._process_watched_dramas(
            watchlist_data, popular_dramas, non_popular_watched_cache
        )
        
        # Save updated non-popular cache
        self.save_non_popular_watched_cache(updated_non_popular_cache)
        
        # Get unwatched dramas (popular dramas minus watched ones)
        unwatched_dramas = self._separate_unwatched_dramas(popular_dramas, watched_slugs)
        
        print(f"Final counts - Watched: {len(watched_dramas)}, Unwatched: {len(unwatched_dramas)}")
        return watched_dramas, unwatched_dramas

    def _extract_crew(self, cast_data: Dict) -> Dict:
        """
        Extract key crew members and funnel them into standardized categories.
        
        Parameters
        ----------
        cast_data : Dict
            Cast and crew data from API containing various roles
            
        Returns
        -------
        Dict
            Dictionary containing:
            - directors: List[str], list of director names
            - screenwriters: List[str], list of screenwriter names  
            - composers: List[str], list of composer names
            
        Notes
        -----
        Handles combined roles like "Screenwriter & Director" by adding 
        the person to both relevant lists. Uses substring matching to 
        categorize roles (e.g., "Assistant Director" goes to directors).
        Removes duplicates from each category.
        """
        if not cast_data or not cast_data.get('data'):
            return {}

        crew = {'directors': [], 'screenwriters': [], 'composers': []}
        casts_section = cast_data.get('data', {}).get('casts', {})

        for role, members in casts_section.items():
            # Get the list of names for the current role
            names = [member.get('name') for member in members if member.get('name')]
            role_lower = role.lower()
            
            # --- NEW: Use separate `if` statements to handle combined roles ---
            
            # Funnel all director-related titles into the 'directors' list
            if 'director' in role_lower:
                crew['directors'].extend(names)
            
            # Funnel all screenwriter-related titles into the 'screenwriters' list
            if 'screenwriter' in role_lower:
                crew['screenwriters'].extend(names)
            
            # Funnel all composer-related titles into the 'composers' list
            if 'composer' in role_lower:
                crew['composers'].extend(names)
        
        # Ensure no duplicates in case a person is listed multiple times
        crew['directors'] = list(set(crew['directors']))
        crew['screenwriters'] = list(set(crew['screenwriters']))
        crew['composers'] = list(set(crew['composers']))
        
        return crew
