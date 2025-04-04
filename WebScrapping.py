import requests
from bs4 import BeautifulSoup
from Bio import Entrez
import pubchempy as pcp
import datetime
import re
import os
import time

CUTOFF_DATE = "2022/01/01"
CUTOFF_YEAR = 2022

primary_keywords = [
    '"polymeric microparticle" AND "drug delivery"',
    '"PLGA" OR "biodegradable polymers"'
]
query = f"({primary_keywords[0]}) AND ({primary_keywords[1]})"
Entrez.email = "your_email@example.com"

def query_google_scholar(query):
    """
    Scrape Google Scholar for relevant papers published after Jan 1, 2022.
    """
    base_url = "https://scholar.google.com/scholar"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {"q": query, "hl": "en", "as_ylo": CUTOFF_YEAR}
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')

    results = []
    for result in soup.find_all('div', class_='gs_ri'):
        title = result.find('h3').text
        link = result.find('a')['href']
        snippet = result.find('div', class_='gs_rs').text if result.find('div', class_='gs_rs') else "No abstract available"
        
        results.append({"title": title, "link": link, "snippet": snippet})
    return results

def query_pubmed(query, max_retries=3):
    """
    Query PubMed using Biopython's Entrez module for papers published after Jan 1, 2022.
    Includes retry logic for handling SSL errors.
    """
    date_filter = f" AND {CUTOFF_DATE}:{datetime.datetime.now().strftime('%Y/%m/%d')}[PDAT]"
    filtered_query = query + date_filter
    
    # Add retry logic
    retry_count = 0
    while retry_count < max_retries:
        try:
            print(f"PubMed query attempt {retry_count+1}...")
            handle = Entrez.esearch(db="pubmed", term=filtered_query, retmax=100)
            record = Entrez.read(handle)
            ids = record["IdList"]
            
            results = []
            for id in ids:
                try:
                    summary = Entrez.esummary(db="pubmed", id=id)
                    summary_record = Entrez.read(summary)
                    title = summary_record[0]["Title"]
                    doi = summary_record[0].get("DOI", "No DOI available")
                    pub_date = summary_record[0].get("PubDate", "Date not available")
                    results.append({"title": title, "doi": doi, "pub_date": pub_date})
                    # Add a small delay between requests to avoid overloading the server
                    time.sleep(0.3)
                except Exception as e:
                    print(f"Error processing PubMed ID {id}: {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            retry_count += 1
            print(f"PubMed query failed (attempt {retry_count}/{max_retries}): {str(e)}")
            if retry_count < max_retries:
                wait_time = 2 * retry_count  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                print("Maximum retries reached. Unable to query PubMed.")
                return []

def query_researchgate(query):
    """
    Scrape ResearchGate for relevant papers published after Jan 1, 2022.
    """
    base_url = "https://www.researchgate.net/search/publication"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {"query": query, "date": f"{CUTOFF_YEAR}-{datetime.datetime.now().year}"}
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')

    results = []
    for result in soup.find_all('div', class_='nova-legacy-v-publication-item__title'):
        title = result.text.strip()
        link = "https://www.researchgate.net" + result.find('a')['href']
        
        year_elem = result.find_next('span', class_='nova-legacy-v-publication-item__meta-data-item')
        pub_year = "Year not found"
        if year_elem:
            pub_year = year_elem.text.strip()
            # Extract year with regex if necessary
            year_match = re.search(r'20\d{2}', pub_year)
            if year_match:
                pub_year = year_match.group(0)
        
        try:
            if int(pub_year) >= CUTOFF_YEAR:
                results.append({"title": title, "link": link, "year": pub_year})
        except ValueError:
            results.append({"title": title, "link": link, "year": pub_year})
            
    return results

def query_nature(query):
    """
    Search Nature publications for papers published after Jan 1, 2022.
    """
    base_url = "https://www.nature.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Format date in Nature's format (YYYY-MM-DD)
    date_from = f"{CUTOFF_YEAR}-01-01"
    
    # Set up search parameters - Nature uses specific format
    params = {
        "q": query, 
        "date_range": date_from, 
        "order": "relevance",
        "article_type": "research"  # Focus on research articles
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    # Nature uses article cards with specific class names
    for article in soup.find_all('article', class_='c-card'):
        try:
            title_elem = article.find('h3', class_='c-card__title')
            title = title_elem.text.strip() if title_elem else "Title not found"
            
            link_elem = article.find('a', 'c-card__link')
            link = "https://www.nature.com" + link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
            
            # Try to find publication date
            date_elem = article.find('time')
            pub_date = date_elem.text.strip() if date_elem else "Date not found"
            
            # Try to find journal/publication name
            journal_elem = article.find('span', class_='c-meta__item')
            journal = journal_elem.text.strip() if journal_elem else "Journal not specified"
            
            results.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "journal": journal
            })
        except Exception as e:
            print(f"Error parsing Nature article: {str(e)}")
    
    return results

def query_mdpi(query):
    """
    Search MDPI (open access publisher) for papers published after Jan 1, 2022.
    """
    base_url = "https://www.mdpi.com/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # MDPI uses year parameter for filtering
    params = {
        "q": query,
        "year_from": CUTOFF_YEAR,
        "year_to": datetime.datetime.now().year,
        "sort": "relevance",
        "view": "default"
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    # MDPI specific article elements
    for article in soup.find_all('div', class_='article-content'):
        try:
            title_elem = article.find('a', class_='title-link')
            title = title_elem.text.strip() if title_elem else "Title not found"
            link = "https://www.mdpi.com" + title_elem['href'] if title_elem and 'href' in title_elem.attrs else "#"
            
            # Find journal name
            journal_elem = article.find('span', class_='journal-name')
            journal = journal_elem.text.strip() if journal_elem else "Journal not specified"
            
            # Find date
            date_elem = article.find('span', class_='date-day')
            pub_date = date_elem.text.strip() if date_elem else "Date not found"
            
            results.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "journal": journal
            })
        except Exception as e:
            print(f"Error parsing MDPI article: {str(e)}")
    
    return results

def query_biorxiv(query):
    """
    Search bioRxiv/medRxiv preprint servers for papers published after Jan 1, 2022.
    """
    base_url = "https://www.biorxiv.org/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Format date in bioRxiv's format
    date_from = f"{CUTOFF_YEAR}-01-01"
    
    params = {
        "terms": query,
        "limit_from": date_from,
        "sort": "relevance-rank"
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    # bioRxiv uses specific article structure
    for article in soup.find_all('div', class_='highwire-article-citation'):
        try:
            title_elem = article.find('span', class_='highwire-cite-title')
            title = title_elem.text.strip() if title_elem else "Title not found"
            
            link_elem = article.find('a', class_='highwire-cite-linked-title')
            link = "https://www.biorxiv.org" + link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
            
            # Find date 
            date_elem = article.find('span', class_='highwire-cite-metadata-date')
            pub_date = date_elem.text.strip() if date_elem else "Date not found"
            
            # Server (bioRxiv or medRxiv)
            server = "bioRxiv/medRxiv"
            
            results.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "journal": server
            })
        except Exception as e:
            print(f"Error parsing bioRxiv article: {str(e)}")
    
    return results

def query_plos(query):
    """
    Search PLOS ONE (open access journal) for papers published after Jan 1, 2022.
    """
    base_url = "https://journals.plos.org/plosone/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Format date for PLOS
    params = {
        "q": query,
        "filterJournals": "PLoSONE",
        "filterStartDate": f"{CUTOFF_YEAR}-01-01",
        "sortOrder": "relevance"
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    # PLOS ONE specific article elements
    for article in soup.find_all('div', class_='search-results-item'):
        try:
            title_elem = article.find('h2', class_='search-results-title')
            title = title_elem.text.strip() if title_elem else "Title not found"
            
            link_elem = title_elem.find('a') if title_elem else None
            link = "https://journals.plos.org" + link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
            
            # Find date
            date_elem = article.find('span', class_='search-results-date')
            pub_date = date_elem.text.strip() if date_elem else "Date not found"
            
            journal = "PLOS ONE"
            
            results.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "journal": journal
            })
        except Exception as e:
            print(f"Error parsing PLOS ONE article: {str(e)}")
    
    return results

def query_frontiers(query):
    """
    Search Frontiers (open access publisher) for papers published after Jan 1, 2022.
    """
    base_url = "https://www.frontiersin.org/search"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    params = {
        "query": query,
        "tab": "articles",
        "publishedSince": f"{CUTOFF_YEAR}-01-01",
        "sortBy": "relevance"
    }
    
    response = requests.get(base_url, headers=headers, params=params)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    results = []
    # Frontiers specific article elements
    for article in soup.find_all('article', class_='SearchResult'):
        try:
            title_elem = article.find('h3', class_='title')
            title = title_elem.text.strip() if title_elem else "Title not found"
            
            link_elem = title_elem.find('a') if title_elem else None
            link = link_elem['href'] if link_elem and 'href' in link_elem.attrs else "#"
            
            # Find date
            date_elem = article.find('span', class_='date')
            pub_date = date_elem.text.strip() if date_elem else "Date not found"
            
            # Find journal
            journal_elem = article.find('span', class_='journal')
            journal = journal_elem.text.strip() if journal_elem else "Frontiers journal"
            
            results.append({
                "title": title,
                "link": link,
                "pub_date": pub_date,
                "journal": journal
            })
        except Exception as e:
            print(f"Error parsing Frontiers article: {str(e)}")
    
    return results

def query_ieee_xplore(query):
    """
    Search IEEE Xplore for papers published after Jan 1, 2022.
    """
    base_url = "https://ieeexplore.ieee.org/search/searchresult.jsp"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Format the query for IEEE Xplore
    formatted_query = query.replace('"', '%22').replace(' ', '+')
    
    # Create parameters for search
    params = {
        "queryText": formatted_query,
        "highlight": "true",
        "returnType": "SEARCH",
        "matchPubs": "true",
        "ranges": f"{CUTOFF_YEAR}01_01{datetime.datetime.now().year}_12_31_Year"
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        # Find paper listings - IEEE has a specific structure
        for paper in soup.select('div.List-results-items'):
            try:
                # Extract title
                title_elem = paper.select_one('h3 a')
                title = title_elem.text.strip() if title_elem else "Title not found"
                
                # Extract link
                paper_id = ""
                if title_elem and 'href' in title_elem.attrs:
                    href = title_elem['href']
                    # Extract article number from URL
                    article_match = re.search(r'document/(\d+)', href)
                    if article_match:
                        paper_id = article_match.group(1)
                
                link = f"https://ieeexplore.ieee.org/document/{paper_id}" if paper_id else "#"
                
                # Extract publication information
                pub_info = paper.select_one('div.description > a')
                journal = pub_info.text.strip() if pub_info else "Journal not specified"
                
                # Extract date
                date_elem = paper.select_one('div.publisher-info-container span.z-950')
                pub_date = date_elem.text.strip() if date_elem else "Date not found"
                
                # Extract DOI if available
                doi = f"10.1109/{paper_id}" if paper_id else "DOI not available"
                
                results.append({
                    "title": title,
                    "link": link,
                    "journal": journal,
                    "pub_date": pub_date,
                    "doi": doi,
                    "paper_id": paper_id
                })
            except Exception as e:
                print(f"Error parsing IEEE paper: {str(e)}")
        
        return results
        
    except Exception as e:
        print(f"Error searching IEEE Xplore: {str(e)}")
        return []

def download_paper(source, paper_data):
    """
    Attempt to download the paper directly to computer when possible.
    """
    print(f"\nAttempting to download: {paper_data['title']}")
    
    # Create a directory for downloads if it doesn't exist
    download_dir = "paper_downloads"
    os.makedirs(download_dir, exist_ok=True)
    
    # Generate a safe filename from the title
    safe_title = "".join(c if c.isalnum() else "_" for c in paper_data['title'])
    safe_title = safe_title[:100]  # Truncate if too long
    file_path = os.path.join(download_dir, f"{safe_title}.pdf")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        if source == "Google Scholar":
            print("Following Google Scholar link...")
            response = requests.get(paper_data['link'], headers=headers)
            
            if response.status_code == 200:
                # Try to find PDF links in the page content
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_links = []
                
                # Look for PDF links with various patterns
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href'].lower()
                    if href.endswith('.pdf') or '/pdf/' in href or 'download' in href and 'pdf' in href:
                        pdf_links.append(a_tag['href'])
                
                if pdf_links:
                    # Try the first PDF link
                    pdf_url = pdf_links[0]
                    if not pdf_url.startswith('http'):
                        # Handle relative URLs
                        if pdf_url.startswith('/'):
                            base_url = '/'.join(paper_data['link'].split('/')[:3])
                            pdf_url = base_url + pdf_url
                        else:
                            pdf_url = paper_data['link'].rsplit('/', 1)[0] + '/' + pdf_url
                    
                    print(f"Found potential PDF link: {pdf_url}")
                    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                    
                    if pdf_response.status_code == 200 and 'application/pdf' in pdf_response.headers.get('Content-Type', ''):
                        with open(file_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Paper downloaded to: {file_path}")
                        return True
                    else:
                        print("Found a PDF link but couldn't download it (may require authentication)")
                else:
                    print("No direct PDF links found on the page")
                
                print(f"Paper link: {paper_data['link']}")
                print("Try accessing this link in your browser with institutional access")
                return False
                
        elif source == "PubMed":
            if paper_data['doi'] != "No DOI available":
                # Try direct DOI resolution first
                doi_url = f"https://doi.org/{paper_data['doi']}"
                print(f"Trying DOI: {doi_url}")
                
                response = requests.get(doi_url, headers=headers, allow_redirects=True)
                
                if response.status_code == 200:
                    # Try to find PDF links in the publisher's page
                    soup = BeautifulSoup(response.content, 'html.parser')
                    pdf_links = []
                    
                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href'].lower()
                        text = a_tag.text.lower()
                        if (href.endswith('.pdf') or '/pdf/' in href or 'download' in href) and ('pdf' in href or 'pdf' in text or 'full text' in text):
                            pdf_links.append(a_tag['href'])
                    
                    if pdf_links:
                        # Try the first PDF link
                        pdf_url = pdf_links[0]
                        if not pdf_url.startswith('http'):
                            # Handle relative URLs
                            if pdf_url.startswith('/'):
                                base_url = '/'.join(response.url.split('/')[:3])
                                pdf_url = base_url + pdf_url
                            else:
                                pdf_url = response.url.rsplit('/', 1)[0] + '/' + pdf_url
                        
                        print(f"Found potential PDF link: {pdf_url}")
                        pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                        
                        if pdf_response.status_code == 200 and 'application/pdf' in pdf_response.headers.get('Content-Type', ''):
                            with open(file_path, 'wb') as f:
                                for chunk in pdf_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            print(f"SUCCESS! Paper downloaded to: {file_path}")
                            return True
                        else:
                            print("Found a PDF link but couldn't download it (likely paywalled)")
                    else:
                        print("No direct PDF links found on the publisher's page")
                
                # If we reach here, we couldn't download directly
                print(f"Direct download not available. Try visiting: {doi_url}")
                print("Note: You may need institutional access or to use a library service.")
                return False
            else:
                print("No DOI available for direct access.")
                return False
                
        elif source == "ResearchGate":
            print("Accessing ResearchGate link...")
            response = requests.get(paper_data['link'], headers=headers)
            
            if response.status_code == 200:
                # Try to find download links on ResearchGate
                soup = BeautifulSoup(response.content, 'html.parser')
                download_links = []
                
                # Look for ResearchGate's download buttons/links
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href'].lower()
                    text = a_tag.text.lower() if a_tag.text else ""
                    if '/download' in href or (('download' in text or 'full-text' in text) and not 'request' in text):
                        download_links.append(a_tag['href'])
                
                if download_links:
                    # Try the first download link
                    dl_url = download_links[0]
                    if not dl_url.startswith('http'):
                        # Handle relative URLs
                        if dl_url.startswith('/'):
                            dl_url = "https://www.researchgate.net" + dl_url
                    
                    print(f"Found potential download link: {dl_url}")
                    dl_response = requests.get(dl_url, headers=headers, stream=True)
                    
                    if dl_response.status_code == 200 and 'application/pdf' in dl_response.headers.get('Content-Type', ''):
                        with open(file_path, 'wb') as f:
                            for chunk in dl_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Paper downloaded to: {file_path}")
                        return True
                    else:
                        print("Found a download link but couldn't access the PDF")
                else:
                    print("No direct download links found on ResearchGate")
                    request_buttons = soup.find_all('button', string=re.compile(r'Request full-text', re.I))
                    if request_buttons:
                        print("This paper requires requesting from the author on ResearchGate")
                
                print(f"Please visit {paper_data['link']} to check download availability.")
                return False
            else:
                print(f"Failed to access ResearchGate page. Status code: {response.status_code}")
                return False
                
        elif source == "Nature":
            print("Accessing Nature publication link...")
            response = requests.get(paper_data['link'], headers=headers)
            if response.status_code == 200:
                print("Nature publication page accessed successfully.")
                
                # Check for PDF download links
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_links = []
                
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href'].lower()
                    text = a_tag.text.lower() if a_tag.text else ""
                    if ('pdf' in href or 'pdf' in text) and ('download' in href or 'download' in text or 'full' in text):
                        pdf_links.append(a_tag['href'])
                
                if pdf_links:
                    # Try to download the first PDF link found
                    pdf_url = pdf_links[0]
                    if not pdf_url.startswith('http'):
                        # Handle relative URLs
                        if pdf_url.startswith('/'):
                            pdf_url = "https://www.nature.com" + pdf_url
                    
                    print(f"Found PDF link: {pdf_url}")
                    print("Attempting to download PDF...")
                    
                    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                    if pdf_response.status_code == 200 and 'application/pdf' in pdf_response.headers.get('Content-Type', ''):
                        with open(file_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Paper downloaded to: {file_path}")
                        return True
                    else:
                        print("Found a PDF link but couldn't download it (likely paywalled)")
                else:
                    print("No direct PDF download link found. Nature articles typically require subscription.")
                    
                print(f"Please visit {paper_data['link']} to check access options.")
                return False
            else:
                print(f"Failed to access Nature page. Status code: {response.status_code}")
                return False
                
        elif source == "MDPI":
            print("Accessing MDPI publication link...")
            response = requests.get(paper_data['link'], headers=headers)
            if response.status_code == 200:
                print("MDPI page accessed successfully.")
                # MDPI is open access, so we should find a PDF download link
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_links = []
                
                # Look for PDF download button/link
                for a_tag in soup.find_all('a', href=True):
                    if 'pdf' in a_tag['href'].lower() or 'download' in a_tag.text.lower():
                        pdf_links.append(a_tag['href'])
                
                if pdf_links:
                    pdf_url = pdf_links[0]
                    if not pdf_url.startswith('http'):
                        pdf_url = "https://www.mdpi.com" + pdf_url
                        
                    print(f"Found PDF link: {pdf_url}")
                    print("Attempting to download PDF...")
                    
                    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                    if pdf_response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Paper downloaded to: {file_path}")
                        return True
                    else:
                        print("Failed to download PDF.")
                else:
                    print("No direct PDF download link found.")
                
                print(f"Please visit {paper_data['link']} for access.")
                return True
            else:
                print(f"Failed to access MDPI page. Status code: {response.status_code}")
                return False

        elif source == "bioRxiv/medRxiv":
            print("Accessing bioRxiv/medRxiv preprint link...")
            response = requests.get(paper_data['link'], headers=headers)
            if response.status_code == 200:
                print("Preprint page accessed successfully.")
                # bioRxiv/medRxiv are open access preprint servers
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_links = []
                
                # bioRxiv PDF links often have specific patterns
                for a_tag in soup.find_all('a', href=True):
                    if '.pdf' in a_tag['href'] or 'download' in a_tag.text.lower():
                        pdf_links.append(a_tag['href'])
                
                if pdf_links:
                    pdf_url = pdf_links[0]
                    if not pdf_url.startswith('http'):
                        pdf_url = "https://www.biorxiv.org" + pdf_url
                        
                    print(f"Found PDF link: {pdf_url}")
                    print("Attempting to download PDF...")
                    
                    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                    if pdf_response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Preprint downloaded to: {file_path}")
                        return True
                    else:
                        print("Failed to download PDF.")
                else:
                    print("No direct PDF download link found.")
                
                print(f"Please visit {paper_data['link']} for access.")
                return True
            else:
                print(f"Failed to access bioRxiv/medRxiv page. Status code: {response.status_code}")
                return False
                
        elif source == "PLOS ONE":
            print("Accessing PLOS ONE publication link...")
            response = requests.get(paper_data['link'], headers=headers)
            if response.status_code == 200:
                print("PLOS ONE page accessed successfully.")
                # PLOS ONE is fully open access
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_links = []
                
                for a_tag in soup.find_all('a', href=True):
                    if 'pdf' in a_tag['href'].lower() or 'download' in a_tag.text.lower():
                        pdf_links.append(a_tag['href'])
                
                if pdf_links:
                    pdf_url = pdf_links[0]
                    if not pdf_url.startswith('http'):
                        pdf_url = "https://journals.plos.org" + pdf_url
                        
                    print(f"Found PDF link: {pdf_url}")
                    print("Attempting to download PDF...")
                    
                    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                    if pdf_response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Paper downloaded to: {file_path}")
                        return True
                    else:
                        print("Failed to download PDF.")
                else:
                    print("No direct PDF download link found.")
                
                print(f"Please visit {paper_data['link']} for access.")
                return True
            else:
                print(f"Failed to access PLOS ONE page. Status code: {response.status_code}")
                return False

        elif source == "Frontiers":
            print("Accessing Frontiers publication link...")
            response = requests.get(paper_data['link'], headers=headers)
            if response.status_code == 200:
                print("Frontiers page accessed successfully.")
                # Frontiers is open access
                soup = BeautifulSoup(response.content, 'html.parser')
                pdf_links = []
                
                for a_tag in soup.find_all('a', href=True):
                    if 'pdf' in a_tag['href'].lower() or ('download' in a_tag.text.lower() and 'article' in a_tag.text.lower()):
                        pdf_links.append(a_tag['href'])
                
                if pdf_links:
                    pdf_url = pdf_links[0]
                    if not pdf_url.startswith('http'):
                        pdf_url = "https://www.frontiersin.org" + pdf_url
                        
                    print(f"Found PDF link: {pdf_url}")
                    print("Attempting to download PDF...")
                    
                    pdf_response = requests.get(pdf_url, headers=headers, stream=True)
                    if pdf_response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            for chunk in pdf_response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"SUCCESS! Paper downloaded to: {file_path}")
                        return True
                    else:
                        print("Failed to download PDF.")
                else:
                    print("No direct PDF download link found.")
                
                print(f"Please visit {paper_data['link']} for access.")
                return True
            else:
                print(f"Failed to access Frontiers page. Status code: {response.status_code}")
                return False
        
        else:
            print(f"Unknown source: {source}")
            return False
            
    except Exception as e:
        print(f"Error trying to download paper: {str(e)}")
        return False

def main():
    print(f"Searching with query: {query}")
    print(f"Filtering for papers published after: January 1, 2022")

    # Query and store all results
    all_sources = []
    
    print("\nQuerying Google Scholar...")
    try:
        google_scholar_results = query_google_scholar(query)
        if google_scholar_results:
            all_sources.append(("Google Scholar", google_scholar_results))
            for i, result in enumerate(google_scholar_results, 1):
                print(f"G{i}. Title: {result['title']}\n   Link: {result['link']}\n   Snippet: {result['snippet']}\n")
        else:
            print("No papers found on Google Scholar for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying Google Scholar: {str(e)}")

    print("\nQuerying PubMed...")
    try:
        pubmed_results = query_pubmed(query)
        if pubmed_results:
            all_sources.append(("PubMed", pubmed_results))
            for i, result in enumerate(pubmed_results, 1):
                print(f"P{i}. Title: {result['title']}\n   DOI: {result['doi']}\n   Publication Date: {result.get('pub_date', 'Date not available')}\n")
        else:
            print("No papers found on PubMed for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying PubMed: {str(e)}")

    print("\nQuerying ResearchGate...")
    try:
        researchgate_results = query_researchgate(query)
        if researchgate_results:
            all_sources.append(("ResearchGate", researchgate_results))
            for i, result in enumerate(researchgate_results, 1):
                print(f"R{i}. Title: {result['title']}\n   Link: {result['link']}\n   Year: {result.get('year', 'Year not found')}\n")
        else:
            print("No papers found on ResearchGate for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying ResearchGate: {str(e)}")
        
    print("\nQuerying Nature Publications...")
    try:
        nature_results = query_nature(query)
        if nature_results:
            all_sources.append(("Nature", nature_results))
            for i, result in enumerate(nature_results, 1):
                print(f"N{i}. Title: {result['title']}\n   Link: {result['link']}\n   Publication: {result['journal']}\n   Date: {result['pub_date']}\n")
        else:
            print("No papers found on Nature for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying Nature: {str(e)}")
        
    # Query MDPI (free open access)
    print("\nQuerying MDPI (Open Access)...")
    try:
        mdpi_results = query_mdpi(query)
        if mdpi_results:
            all_sources.append(("MDPI", mdpi_results))
            for i, result in enumerate(mdpi_results, 1):
                print(f"M{i}. Title: {result['title']}\n   Link: {result['link']}\n   Journal: {result['journal']}\n   Date: {result['pub_date']}\n")
        else:
            print("No papers found on MDPI for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying MDPI: {str(e)}")
        
    # Query bioRxiv/medRxiv (free preprint server)
    print("\nQuerying bioRxiv/medRxiv (Preprints)...")
    try:
        biorxiv_results = query_biorxiv(query)
        if biorxiv_results:
            all_sources.append(("bioRxiv/medRxiv", biorxiv_results))
            for i, result in enumerate(biorxiv_results, 1):
                print(f"B{i}. Title: {result['title']}\n   Link: {result['link']}\n   Server: {result['journal']}\n   Date: {result['pub_date']}\n")
        else:
            print("No papers found on bioRxiv/medRxiv for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying bioRxiv/medRxiv: {str(e)}")
        
    # Query PLOS ONE (free open access)
    print("\nQuerying PLOS ONE (Open Access)...")
    try:
        plos_results = query_plos(query)
        if plos_results:
            all_sources.append(("PLOS ONE", plos_results))
            for i, result in enumerate(plos_results, 1):
                print(f"L{i}. Title: {result['title']}\n   Link: {result['link']}\n   Journal: {result['journal']}\n   Date: {result['pub_date']}\n")
        else:
            print("No papers found on PLOS ONE for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying PLOS ONE: {str(e)}")
        
    # Query Frontiers (free open access)
    print("\nQuerying Frontiers (Open Access)...")
    try:
        frontiers_results = query_frontiers(query)
        if frontiers_results:
            all_sources.append(("Frontiers", frontiers_results))
            for i, result in enumerate(frontiers_results, 1):
                print(f"F{i}. Title: {result['title']}\n   Link: {result['link']}\n   Journal: {result['journal']}\n   Date: {result['pub_date']}\n")
        else:
            print("No papers found on Frontiers for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying Frontiers: {str(e)}")
        
        # Add this section in the main() function where the other sources are queried
    print("\nQuerying IEEE Xplore...")
    try:
        ieee_results = query_ieee_xplore(query)
        if ieee_results:
            all_sources.append(("IEEE Xplore", ieee_results))
            for i, result in enumerate(ieee_results, 1):
                print(f"I{i}. Title: {result['title']}\n   Link: {result['link']}\n   Journal: {result['journal']}\n   Date: {result['pub_date']}\n   DOI: {result['doi']}\n")
        else:
            print("No papers found on IEEE Xplore for this topic since January 1, 2022.")
    except Exception as e:
        print(f"Error querying IEEE Xplore: {str(e)}")
    
    # Interactive interface for paper selection
    if all_sources:
        print("\n" + "="*80)
        print("PAPER DOWNLOAD INTERFACE")
        print("="*80)
        print("Enter the paper ID you want to download (e.g., G1 for first Google Scholar result)")
        print("Multiple IDs with commas supported (e.g., G1,P2,R3,N1)")
        print("Enter 'exit' to quit")
        
        while True:
            user_input = input("\nWhich paper(s) would you like to download? ").strip()
            
            if user_input.lower() == 'exit':
                print("Exiting download interface.")
                break
                
            paper_ids = [pid.strip() for pid in user_input.split(',')]
            for paper_id in paper_ids:
                try:
                    # Parse source and index
                    if not paper_id or len(paper_id) < 2:
                        print(f"Invalid ID format: {paper_id}. Use format like G1, P2, R3.")
                        continue
                        
                    source_letter = paper_id[0].upper()
                    index_str = paper_id[1:]
                    
                    # Map letter to source - UPDATED to include all sources
                    source_map = {
                        'G': 'Google Scholar', 
                        'P': 'PubMed', 
                        'R': 'ResearchGate', 
                        'N': 'Nature',
                        'M': 'MDPI',
                        'B': 'bioRxiv/medRxiv',
                        'L': 'PLOS ONE',
                        'F': 'Frontiers'
                    }
                    
                    if source_letter not in source_map:
                        print(f"Invalid source letter in {paper_id}. Use G/P/R/N/M/B/L/F for sources.")
                        continue
                        
                    source_name = source_map[source_letter]
                    
                    # Find the source in our results
                    source_found = False
                    for src, results in all_sources:
                        if src == source_name:
                            source_found = True
                            try:
                                index = int(index_str) - 1
                                if 0 <= index < len(results):
                                    # Try to download the paper
                                    success = download_paper(src, results[index])
                                    if success:
                                        print("Download attempt complete. Check paper_downloads folder.")
                                    else:
                                        print("Unable to download paper automatically. You may need to access it manually.")
                                else:
                                    print(f"Index {index+1} out of range for {src}. Max index is {len(results)}.")
                            except ValueError:
                                print(f"Invalid index: {index_str}. Please use a number after the source letter.")
                            break
                            
                    if not source_found:
                        print(f"Source {source_name} not found in results.")
                        
                except Exception as e:
                    print(f"Error processing {paper_id}: {str(e)}")
            
            another = input("Do you want to download more papers? (y/n): ").strip().lower()
            if another != 'y':
                print("Exiting download interface.")
                break
    else:
        print("\nNo results found from any source. Cannot download papers.")

if __name__ == "__main__":
    main()