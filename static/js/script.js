// TODO: обновить все до 4 шага
// TODO нормальное отображение ошибок
// TODO при возвращении на конкретный шаг не обнуляется
// TODO разбить по файлам алгоритм

let currentProcess = null;
let selectedFileType = null;
let stepCounts = 4;

const fileInput = document.getElementById('fileInput');
  const fileNameSpan = document.getElementById('fileName');

  fileInput.addEventListener('change', function () {
    if (fileInput.files.length > 0) {
      fileNameSpan.textContent = fileInput.files[0].name;
    }
    document.getElementById('uploadStatus').textContent = '';
    document.getElementById('resetStep2').classList.add('hidden');
    document.getElementById('uploadBtn').classList.remove('inactive');
  });

document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    selectedFileType = file.name.split(".").pop().toLowerCase();
    if (selectedFileType !== 'csv' && selectedFileType !== 'pdf') {
        document.getElementById('uploadStatus').textContent = 'Ошибка: неверный тип файла. Пожалуйста, выберите файл с расширением .csv или .pdf';
        return;
    }
    
    fetch('/create-process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_type: selectedFileType })
    })
    .then(response => response.json())
    .then(data => {
        currentProcess = data.id;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('process_id', currentProcess);
        
        fetch('/upload-file', { method: 'POST', body: formData })
        .then(response => response.json())
        .then(data => {

            document.getElementById('uploadStatus').textContent = 'Файл успешно загружен!';
            document.getElementById('step2').classList.add('complete');
            document.getElementById('step2').classList.add('inactive');
            document.getElementById('resetStep2').classList.remove('hidden');
            document.getElementById('step3').classList.remove('inactive');
        })
        .catch(error => {
            document.getElementById('uploadStatus').textContent = 'Ошибка при загрузке файла: ' + error;
        });
    });
});

function triggerUpload() {
    var form = document.getElementById('uploadForm');
    if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
    } else {
        form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
    }
}

// Шаг 3
document.getElementById('processBtn').addEventListener('click', function() {
    document.getElementById('resetStep2').classList.add('inactive');

    document.getElementById('processBtn').classList.add('hidden');
    document.getElementById('processingStatus').textContent = 'Инициализрована обработка.';
    fetch(`/process/${currentProcess}`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        document.getElementById('processingStatus').textContent = data.message;
        pollProcessingStatus();
    })
    .catch(error => {
        document.getElementById('processingStatus').textContent = 'Ошибка обработки';
        document.getElementById('processingError').textContent = error;
        document.getElementById('processingError').classList.remove('hidden');
    });
});

// для опрашивания статуса обработки (3 шага)
function pollProcessingStatus() {
    const statusElement = document.getElementById('processingStatus');
    const errorElement = document.getElementById('processingError');

    const checkStatus = setInterval(() => {
        fetch(`/status/${currentProcess}`)
        .then(response => response.json())
        .then(data => {
            statusElement.textContent = data.status;

            if (data.status === 'ошибка' || data.status === 'обработка завершена' || data.current_step >= 3) {
                document.getElementById('resetStep2').classList.remove('inactive');
                clearInterval(checkStatus);

                if (data.status === 'ошибка') {
                    errorElement.textContent = data.error_message;
                    errorElement.classList.remove('hidden');
                }
                else {
                    statusElement.textContent = 'Обработка файла завершена!';
                    if (selectedFileType === 'pdf') {
                        document.getElementById('downloadBtn').classList.remove('hidden');
                    }
                    document.getElementById('step3').classList.add('complete');
                    document.getElementById('step4').classList.remove('inactive');
                }
            }
        });
    }, 1000);
}

// для скачивания
document.getElementById('downloadBtn').addEventListener('click', function() {
    const downloadUrl = `/download/${currentProcess}`;
    
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

// Шаг 4
document.getElementById('paramsForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const params = {
        checkbox1: document.getElementById('option1').checked,
        checkbox2: document.getElementById('option2').checked,
        checkbox3: document.getElementById('option3').checked,
        checkbox4: document.getElementById('option4').checked,
        checkbox5: document.getElementById('option5').checked,
        checkbox6: document.getElementById('option6').checked,
        checkbox7: document.getElementById('option7').checked,
        parts_count: Number(document.getElementById('partsCount').value),
        threshold: Number(document.getElementById('threshold').value)
    };

    fetch(`/set-params/${currentProcess}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('step4').classList.add('complete');
        document.getElementById('step4').classList.add('inactive');
        document.getElementById('resetStep4').classList.remove('hidden');
        document.getElementById('step5').classList.remove('inactive');
    })
    .catch(error => {
        alert('Ошибка установки параметров: ' + error);
    });
});

function toggleHiddenParams() {
    const hiddenParamsDiv = document.getElementById("hiddenParams");
    const toggleButton = document.getElementById("toggleHiddenParamsBtn");
    if (hiddenParamsDiv.style.display === "none") {
        hiddenParamsDiv.style.display = "block";
        toggleButton.textContent = "Спрятать дополнительные параметры";
    } else {
        hiddenParamsDiv.style.display = "none";
        toggleButton.textContent = "Показать дополнительные параметры";
    }
}
document.getElementById('partsCount').addEventListener('keydown', function(event) {
    event.preventDefault();
});
document.getElementById('threshold').addEventListener('keydown', function(event) {
    event.preventDefault();
});
document.getElementById('partsCount').addEventListener('paste', function(event) {
    event.preventDefault();
});
document.getElementById('threshold').addEventListener('paste', function(event) {
    event.preventDefault();
});

function triggerParamsSubmit() {
    var form = document.getElementById('paramsForm');
    if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
    } else {
        form.dispatchEvent(new Event('submit', {bubbles: true, cancelable: true}));
    }
}

// Шаг 5
document.getElementById('analyzeBtn').addEventListener('click', function() {
    document.getElementById('resetStep2').classList.add('inactive');
    document.getElementById('resetStep4').classList.add('inactive');

    document.getElementById('analyzeBtn').classList.add('hidden');
    document.getElementById('analysisStatus').textContent = 'Инициализирован анализ.';

    fetch(`/analyze/${currentProcess}`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        document.getElementById('analysisStatus').textContent = data.message;
        pollAnalysisStatus();
    })
    .catch(error => {
        document.getElementById('analysisStatus').textContent = 'Ошибка анализа';
        document.getElementById('analysisError').textContent = error;
        document.getElementById('analysisError').classList.remove('hidden');
    });
});

// для опрашивания статуса анализа (5 шага)
function pollAnalysisStatus() {
    const analysisStatusElement = document.getElementById('analysisStatus');
    const analysisErrorElement = document.getElementById('analysisError');

    const checkAnalysisStatus = setInterval(() => {
        fetch(`/status/${currentProcess}`)
        .then(response => response.json())
        .then(data => {
            analysisStatusElement.textContent = data.status;
            
            if (data.status === 'ошибка' || data.status === 'анализ завершен' || data.current_step >= 5) {
                document.getElementById('resetStep2').classList.remove('inactive');
                document.getElementById('resetStep4').classList.remove('inactive');
                clearInterval(checkAnalysisStatus);

                if (data.status === 'ошибка') {
                    analysisErrorElement.textContent = data.error_message;
                    analysisErrorElement.classList.remove('hidden');
                } else {
                    analysisStatusElement.textContent = 'Анализ завершён!';
                    
                    document.getElementById('analysisResults').innerHTML = `
                        <div class="analysis-container">
                            <div class="analysis-image">
                                <img src="${data.results.image_data}" style="max-width: 100%; height: auto;">
                            </div>
                            <div class="analysis-details">
                                ${data.results.summary}
                            </div>
                        </div>
                    `;
                    document.getElementById('step5').classList.add('complete');
                }
            }
        });
    }, 1000);
}

// для обнуления полей при перезагрузке
window.addEventListener('load', function() {
    currentProcess = null;
    resetManySteps(1);
});

// для обнуления одного конкретного шага
function resetOneStep(step) {
    switch (step) {
        case 1:
            selectedFileType = null;
            document.getElementById('step2').classList.remove('complete');
            document.getElementById('step2').classList.remove('inactive');
            document.getElementById('uploadStatus').textContent = '';
            document.getElementById('fileInput').value = '';
            document.getElementById('uploadBtn').classList.add('inactive');
            document.getElementById('fileName').textContent = '';
            document.getElementById('resetStep2').classList.add('hidden');
            document.getElementById('step3').classList.add('inactive');
            break;
        case 2:
            document.getElementById('step3').classList.remove('complete');
            document.getElementById('processingStatus').textContent = '';
            document.getElementById('processingError').classList.add('hidden');
            document.getElementById('downloadBtn').classList.add('hidden');
            document.getElementById('processBtn').classList.remove('hidden');
            document.getElementById('step4').classList.add('inactive');
            break;
        case 3:
            document.querySelectorAll('#paramsForm input[type="checkbox"]').forEach(checkbox => {
                checkbox.checked = false;
            });
            document.getElementById('partsCount').value = "13";
            document.getElementById('threshold').value = "10";
            document.getElementById('step4').classList.remove('complete');
            document.getElementById('step4').classList.remove('inactive');
            document.getElementById('resetStep4').classList.add('hidden');
            document.getElementById('step5').classList.add('inactive');
            break;
        case 4:
            document.getElementById('step5').classList.remove('complete');
            document.getElementById('analyzeBtn').classList.remove('hidden');
            document.getElementById('analysisResults').innerHTML = '';
            document.getElementById('analysisStatus').textContent = '';
            document.getElementById('analysisError').classList.add('hidden');
            break;
    }
}

// для каскадного обнуления шагов
function resetManySteps(step) {
    for (let i = stepCounts; i >= step; i--) {
        resetOneStep(i);
    }
}