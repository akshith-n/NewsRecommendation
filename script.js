let articles = [];
let liked_content = [];

const BACKEND_URL = "https://newsrecommendation-2.onrender.com";


async function getmore() {
    const dp = document.getElementById("more");
    dp.style.backgroundColor = dp.style.backgroundColor === "red" ? "black" : "red";

    try {
        // Fetch recommendation data based on liked content
        const response = await fetch(`${BACKEND_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ liked_news: liked_content })
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        // Process response and update UI with recommended articles
        const data = await response.json();
        const newsContainer = document.getElementById('news-container');
        console.log('Recommended articles:', data.recommended_articles);
        let recommended_articles = data.recommended_articles;
        newsContainer.innerHTML = '';

        recommended_articles.forEach((articleData, index) => {
            if (index >= liked_content.length) { 
                const articleElement = createArticle(articleData);
                newsContainer.appendChild(articleElement);
            }
        });
        recommended_articles.length = 0;

    } catch (error) {
        console.error('There has been a problem with your fetch operation:', error);
    }
}

async function fetchData() {
    try {
        // Fetch initial data from the server
        const response = await fetch(`${BACKEND_URL}/`);
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching data:', error);
        return null;
    }
}

document.addEventListener('DOMContentLoaded', async () => {
    const newsContainer = document.getElementById('news-container');

    try {
        // Fetch data on page load and initialize the articles list
        const fetchedData = await fetchData();
        console.log("Fetched data:", fetchedData);
        if (fetchedData) {
            articles = fetchedData;
            console.log('Fetched articles:', articles);

            // Create and append articles to the news container
            articles.forEach(articleData => {
                const articleElement = createArticle(articleData);
                newsContainer.appendChild(articleElement);
            });

        } else {
            console.log('Failed to fetch articles.');
        }
    } catch (error) {
        console.error('Error initializing:', error);
    }

    // Create scroll-to-top button and handle its behavior
    const scrollTopButton = createScrollTopButton();
    document.body.appendChild(scrollTopButton);

    window.addEventListener('scroll', () => {
        // Show or hide scroll-to-top button based on scroll position
        scrollTopButton.style.display = window.scrollY > 200 ? 'block' : 'none';
    });
});

// Function to create an article element with like button
function createArticle(content) {
    const article = document.createElement('article');
    const articleContent = document.createElement('div');
    articleContent.classList.add('article-content');
    articleContent.innerHTML = `<h2>${content}</h2>`;

    const likeButton = document.createElement('button');
    likeButton.textContent = 'Like';
    likeButton.addEventListener('click', () => {
        likeArticle(content);
    });

    article.appendChild(articleContent);
    article.appendChild(likeButton);

    return article;
}

// Function to handle like button click event
function likeArticle(content) {
    console.log("Like button clicked:", content);
    liked_content.push(content);
    const index = articles.indexOf(content);
    if (index !== -1) {
        articles.splice(index, 1);
    }
}

// Function to create and configure the scroll-to-top button
function createScrollTopButton() {
    const button = document.createElement('button');
    button.textContent = 'Scroll to Top';
    button.classList.add('scroll-top-button');
    button.style.position = 'fixed';
    button.style.bottom = '20px';
    button.style.right = '20px';
    button.style.display = 'none';

    button.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    return button;
}
