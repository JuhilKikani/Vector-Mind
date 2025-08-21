document.addEventListener('DOMContentLoaded', function() {
    const uploadButton = document.getElementById('uploadButton');
    const pdfFile = document.getElementById('pdfFile');
    const uploadStatus = document.getElementById('uploadStatus');
    const uploadLoader = document.getElementById('uploadLoader');

    const askButton = document.getElementById('askButton');
    const questionInput = document.getElementById('questionInput');
    const answerDisplay = document.getElementById('answerDisplay');
    const answerLoader = document.getElementById('answerLoader');

    // Function to show/hide loading indicators
    function showLoader(loaderElement) {
        loaderElement.style.display = 'block';
    }

    function hideLoader(loaderElement) {
        loaderElement.style.display = 'none';
    }

    // Event listener for PDF upload
    uploadButton.addEventListener('click', async function() {
        const file = pdfFile.files[0];
        if (!file) {
            uploadStatus.textContent = 'Please select a PDF file to upload.';
            uploadStatus.className = 'mt-3 text-sm text-center text-red-500';
            return;
        }

        uploadStatus.textContent = 'Uploading and analyzing document... This may take a moment.';
        uploadStatus.className = 'mt-3 text-sm text-center text-gray-600';
        showLoader(uploadLoader);
        
        const formData = new FormData();
        formData.append('pdfFile', file);

        try {
            const response = await fetch('/upload_pdf', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                uploadStatus.textContent = data.message;
                uploadStatus.className = 'mt-3 text-sm text-center text-green-600';
                answerDisplay.innerHTML = '<p>Document processed! You can now ask questions related to it.</p>';
            } else {
                uploadStatus.textContent = `Error: ${data.message}`;
                uploadStatus.className = 'mt-3 text-sm text-center text-red-500';
            }
        } catch (error) {
            console.error('Error:', error);
            uploadStatus.textContent = 'An unexpected error occurred during upload. Please try again.';
            uploadStatus.className = 'mt-3 text-sm text-center text-red-500';
        } finally {
            hideLoader(uploadLoader);
        }
    });

    // Event listener for asking a question
    askButton.addEventListener('click', async function() {
        const question = questionInput.value.trim();
        if (!question) {
            answerDisplay.innerHTML = '<p class="text-red-500">Please enter a question.</p>';
            return;
        }

        answerDisplay.innerHTML = '<p class="text-gray-600">Searching for answers...</p>';
        showLoader(answerLoader);

        try {
            const response = await fetch('/ask_question', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            if (response.ok) {
                answerDisplay.innerHTML = `<p>${data.answer}</p>`;
            } else {
                answerDisplay.innerHTML = `<p class="text-red-500">Error: ${data.message}</p>`;
            }
        } catch (error) {
            console.error('Error:', error);
            answerDisplay.innerHTML = '<p class="text-red-500">An unexpected error occurred while getting the answer. Please try again.</p>';
        } finally {
            hideLoader(answerLoader);
        }
    });
});
