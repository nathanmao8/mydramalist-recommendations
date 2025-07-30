import os
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Set
import time
import random

class DataLoader:
    def __init__(self, api):
        self.api = api
        self.cache_file = 'popular_dramas_processed.json'
        self.non_popular_cache_file = 'watched_non_popular_cache.json'
        self.failed_dramas_file = 'failed_dramas.json'

    def get_popular_drama_slugs(self) -> Set[str]:
        """
        Extract drama slugs from all popular*.html files in the specified folder. 
        Files contain the 1000 most popular dramas and 500 most popular movies based on MyDramaList.com's rankings on July 17, 2025. 
                    
        Returns:
            set: All drama slugs found across all files
        """
        folder_path = 'html_popular'
        all_slugs = set()
        
        # Loop through all 75 files
        for i in range(1, 76):
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
        """Load user's rated dramas from watchlist and return watched slugs."""
        watchlist_data = self.api.get_user_dramalist(user_id)
        watched_slugs = set()
        
        if watchlist_data and watchlist_data.get('data'):
            # Access the 'list' within data, then 'Completed' items
            drama_list = watchlist_data.get('data', {}).get('list', {})
            completed_items = drama_list.get('Completed', {}).get('items', [])
            
            for item in completed_items:
                slug = item.get('id', '')
                if slug:
                    watched_slugs.add(slug)
        
        return watchlist_data, watched_slugs
    
    def load_drama_details(self, slug: str) -> Dict:
        """Load comprehensive drama information."""
        drama_info = self.api.get_drama_info(slug)
        cast_info = self.api.get_cast(slug)
        reviews_info = self.api.get_reviews(slug)
        
        return {
            'info': drama_info,
            'cast': cast_info,
            'reviews': reviews_info
        }
    
    def extract_and_clean_drama_features(self, drama_details: Dict, text_processor) -> Dict:
        """Extract and clean necessary features from drama details."""
        try:
            # Extract basic info - note the nested structure
            drama_info = drama_details.get('info', {}).get('data', {})
            title = drama_info.get('title', '')
            slug = drama_details.get('info', {}).get('slug_query', '')
            
            # Extract and clean synopsis, remove "(Source: )" text
            raw_synopsis = drama_info.get('synopsis', '')
            splits = raw_synopsis.split('\n(')
            synopsis = splits[0]
            synopsis_clean = text_processor.clean_text(synopsis)
            
            # Extract and clean reviews
            reviews_combined, helpfulness_weights = text_processor.process_reviews(drama_details.get('reviews', {}))
            
            # Extract genres, tags, and cast from 'others' section
            others_section = drama_info.get('others', {})
            genres = others_section.get('genres', [])
            raw_tags = others_section.get('tags', [])
            # Handle extra text at end of last tag'
            last = raw_tags[-1:][0][:-12]
            tags = raw_tags[:-1]
            tags.append(last)

            # Extract main cast
            main_cast = self.extract_main_cast(drama_details.get('cast', {}))

            # Extract crew
            crew_data = self._extract_crew(drama_details.get('cast', {}))

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
                'sentiment_features': sentiment_features
            }
        except Exception as e:
            print(f"Error processing drama {slug}: {e}")
            return None
    
    def extract_main_cast(self, cast_data: Dict) -> List[str]:
        """Extract main role cast members."""
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
    
    def create_popular_dramas_cache(self, text_processor, retry_on_error=True, max_retries=3) -> Dict:
        """Create cache with built-in automatic retry mechanism."""
        
        popular_slugs = self.get_popular_drama_slugs()
        processed_dramas = {}
        failed_dramas = []
        
        # First pass: process all dramas
        for i, slug in enumerate(popular_slugs, 1):
            try:
                print(f"Processing drama {i}/{len(popular_slugs)}: {slug}")
                
                # Add small delay to avoid overwhelming the API
                time.sleep(0.5)
                
                drama_details = self.load_drama_details(slug)
                processed_drama = self.extract_and_clean_drama_features(drama_details, text_processor)
                
                if processed_drama:
                    processed_dramas[slug] = processed_drama
                else:
                    failed_dramas.append(slug)
                    
            except Exception as e:
                print(f"Failed to process {slug}: {e}")
                failed_dramas.append(slug)
        
        # Automatic retry mechanism for failed dramas
        if retry_on_error and failed_dramas:
            print(f"\nStarting automatic retry for {len(failed_dramas)} failed dramas...")
            
            retry_count = 0
            while failed_dramas and retry_count < max_retries:
                retry_count += 1
                print(f"Retry attempt {retry_count}/{max_retries}")
                
                current_failures = failed_dramas.copy()
                failed_dramas.clear()
                
                for i, slug in enumerate(current_failures, 1):
                    try:
                        print(f"Retrying drama {i}/{len(current_failures)}: {slug}")
                        
                        # Exponential backoff with randomization
                        delay = random.uniform(1, 3) * retry_count
                        time.sleep(delay)
                        
                        drama_details = self.load_drama_details(slug)
                        processed_drama = self.extract_and_clean_drama_features(drama_details, text_processor)
                        
                        if processed_drama:
                            processed_dramas[slug] = processed_drama
                            print(f"Successfully processed {slug} on retry {retry_count}")
                        else:
                            failed_dramas.append(slug)
                            
                    except Exception as e:
                        print(f"Retry failed for {slug}: {e}")
                        failed_dramas.append(slug)
                
                if failed_dramas:
                    print(f"Still have {len(failed_dramas)} failed dramas after retry {retry_count}")
                    if retry_count < max_retries:
                        print(f"Waiting 30 seconds before next retry...")
                        time.sleep(30)  # Longer wait between retry rounds
        
        # Save results
        self._save_cache_results(processed_dramas, failed_dramas)
        
        # Report final results
        self._report_final_results(processed_dramas, failed_dramas, len(popular_slugs))
        
        return processed_dramas
    
    def _save_cache_results(self, processed_dramas: Dict, failed_dramas: List):
        """Save processed dramas and failed dramas list."""
        
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
        """Print comprehensive results report."""
        
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

    def load_popular_dramas_cache(self) -> Dict:
        """Load popular dramas from cache file."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    
    def load_non_popular_watched_cache(self) -> Dict:
        """Loads the cache of non-popular watched dramas from its JSON file."""
        if not os.path.exists(self.non_popular_cache_file):
            return {}
        try:
            with open(self.non_popular_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def save_non_popular_watched_cache(self, cache_data: Dict):
        """Saves the updated non-popular watched cache to its JSON file."""
        with open(self.non_popular_cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)

    def load_all_drama_data(self, user_id: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load all watched and unwatched drama data with a dual-caching system.
        - One cache for the 1000 popular dramas.
        - A second, persistent cache for non-popular dramas from the user's watchlist.
        """
        
        # Load user's watchlist and get watched slugs
        watchlist_data, watched_slugs = self.load_user_watchlist(user_id)
        print(f"User has watched {len(watched_slugs)} dramas")
        
        # Load or create popular dramas cache
        if os.path.exists(self.cache_file):
            print(f"Loading popular dramas from cache: {self.cache_file}")
            popular_dramas = self.load_popular_dramas_cache()
        else:
            print("Cache file not found. Creating cache...")
            # Import here to avoid circular imports
            from text_processor import TextProcessor
            text_processor = TextProcessor()
            popular_dramas = self.create_popular_dramas_cache(text_processor)
        
        print(f"Loaded {len(popular_dramas)} popular dramas from cache")
        
        non_popular_watched_cache = self.load_non_popular_watched_cache()
        print(f"Loaded {len(non_popular_watched_cache)} dramas from the non-popular watched cache.")

        # Separate watched and unwatched dramas
        watched_dramas = []
        unwatched_dramas = []
        
        # Process watched dramas (need to add ratings)
        completed_items = watchlist_data.get('data', {}).get('list', {}).get('Completed', {}).get('items', [])
        
        for item in completed_items:
            slug = item.get('id', '')
            score = item.get('score', '0.0')
            
            try:
                rating = float(score)
            except (ValueError, TypeError):
                print(f"Invalid rating for {slug}: {score}")
                continue
            
            processed_drama = None
            
            # Check the popular cache first
            if slug in popular_dramas:
                processed_drama = popular_dramas[slug].copy()
            # If not found, check the non-popular cache
            elif slug in non_popular_watched_cache:
                processed_drama = non_popular_watched_cache[slug].copy()
            # If in neither cache, this is a new, non-popular drama that must be fetched
            else:
                print(f"Fetching new non-popular watched drama: {slug}")
                try:
                    drama_details = self.load_drama_details(slug)
                    from text_processor import TextProcessor # Avoid circular import
                    text_processor = TextProcessor()
                    processed_drama = self.extract_and_clean_drama_features(drama_details, text_processor)
                    
                    # If fetched successfully, add it to our in-memory cache
                    if processed_drama:
                        non_popular_watched_cache[slug] = processed_drama
                except Exception as e:
                    print(f"Failed to fetch details for new drama {slug}: {e}")

            # If we have the drama data (from any source), add the rating and append it
            if processed_drama:
                processed_drama['user_rating'] = rating
                watched_dramas.append(processed_drama)

        # After processing all watched dramas, save the potentially updated non-popular cache
        self.save_non_popular_watched_cache(non_popular_watched_cache)
        print(f"Saved non-popular cache with {len(non_popular_watched_cache)} dramas.")

        # The logic for unwatched dramas remains the same: they are the popular dramas minus the watched ones
        for slug, drama_data in popular_dramas.items():
            if slug not in watched_slugs:
                unwatched_dramas.append(drama_data)
        
        print(f"Final counts - Watched: {len(watched_dramas)}, Unwatched: {len(unwatched_dramas)}")
        return watched_dramas, unwatched_dramas

    def _extract_crew(self, cast_data: Dict) -> Dict:
        """
        Extracts key crew members and funnels them into standardized categories.
        Handles roles like "Screenwriter & Director" by adding the person to both lists.
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
