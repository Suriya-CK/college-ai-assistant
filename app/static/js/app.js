/**
 * College AI Voice Assistant Frontend Controller
 */

let activeCollegeId = null;
let activeCollege = null;
let activeConversationId = null;
let currentEventSource = null;
let extractionPollTimer = null;
let appLanguages = {};

// Example questions for English; other languages use the translated ones from /api/languages
const QUICK_QUESTIONS_EN = [
    { icon: "briefcase", color: "text-teal-400", label: "Companies hiring here", question: "Which companies hire students from this college?" },
    { icon: "graduation-cap", color: "text-blue-400", label: "UG courses offered", question: "List all the UG courses offered here." },
    { icon: "book-open", color: "text-indigo-400", label: "PG courses offered", question: "List all the PG courses offered here." },
    { icon: "indian-rupee", color: "text-emerald-400", label: "Fees & scholarships", question: "What are the fees and scholarships available?" },
    { icon: "clipboard-check", color: "text-amber-400", label: "Admission & eligibility", question: "What is the admission process and eligibility?" },
    { icon: "building-2", color: "text-rose-400", label: "Hostel & facilities", question: "What hostel and campus facilities are available?" },
];

document.addEventListener("DOMContentLoaded", async () => {
    initUI();
    initVoiceIntegration();
    setupEventListeners();
    await loadLanguages();
    await applyLanguage(currentLanguage());
    checkAISettings();

    renderSuggestions();

    // Reopen the college used last time (if it still exists)
    let lastId = null;
    try { lastId = parseInt(localStorage.getItem("lastCollegeId")); } catch (e) { /* storage unavailable */ }
    if (lastId && document.querySelector(`.college-card[data-college-id="${lastId}"]`)) {
        selectCollege(lastId);
    }
});

function currentLanguage() {
    const langSelect = document.getElementById("languageSelect");
    return langSelect ? langSelect.value : "en";
}

function getQuickQuestions() {
    const lang = currentLanguage();
    const translated = (appLanguages[lang] && appLanguages[lang].sample_questions) || [];
    if (lang === "en" || lang === "auto" || translated.length === 0) return QUICK_QUESTIONS_EN;
    return translated.map((q, i) => ({ ...QUICK_QUESTIONS_EN[i % QUICK_QUESTIONS_EN.length], label: q, question: q }));
}

function askQuickQuestion(question) {
    if (!activeCollegeId) {
        fillPrompt(question);
        if (window.innerWidth < 768) openSidebar();
        return;
    }
    submitQuestion(question);
}

// Renders example questions into the welcome card grid and the chip row under the college banner
function renderSuggestions() {
    const questions = getQuickQuestions();
    document.querySelectorAll(".suggestion-grid").forEach((grid) => {
        grid.innerHTML = "";
        questions.forEach((q) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "p-2.5 rounded-xl border border-slate-700/70 bg-slate-800/70 hover:border-teal-500/50 text-xs text-slate-300 hover:text-teal-300 transition flex items-center gap-2 text-left";
            btn.innerHTML = `<i data-lucide="${q.icon}" class="w-4 h-4 ${q.color} flex-shrink-0"></i><span class="truncate">${escapeHtml(q.label)}</span>`;
            btn.addEventListener("click", () => askQuickQuestion(q.question));
            grid.appendChild(btn);
        });
    });
    document.querySelectorAll(".suggestion-row").forEach((row) => {
        row.innerHTML = "";
        questions.forEach((q) => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "suggestion-chip";
            btn.innerHTML = `<i data-lucide="${q.icon}" class="w-3.5 h-3.5 ${q.color}"></i><span>${escapeHtml(q.label)}</span>`;
            btn.addEventListener("click", () => askQuickQuestion(q.question));
            row.appendChild(btn);
        });
    });
    lucide.createIcons();
}

function openSidebar() {
    document.getElementById("sidebar").classList.add("open");
    document.getElementById("sidebarBackdrop").classList.remove("hidden");
}

function closeSidebar() {
    document.getElementById("sidebar").classList.remove("open");
    document.getElementById("sidebarBackdrop").classList.add("hidden");
}

async function loadLanguages() {
    try {
        const resp = await fetch("/api/languages");
        appLanguages = await resp.json();
    } catch (e) {
        console.warn("Could not load languages:", e);
    }
}

async function checkAISettings() {
    try {
        const resp = await fetch("/api/settings");
        const data = await resp.json();
        const badge = document.getElementById("aiStatusBtn");
        const dot = document.getElementById("aiStatusDot");
        const text = document.getElementById("aiStatusText");
        const banner = document.getElementById("aiKeyNoticeBanner");

        const names = { groq: "Groq", openai: "OpenAI", gemini: "Gemini", openrouter: "OpenRouter" };
        const activeHasKey = data.active_provider && data.active_provider !== "fallback";
        if (activeHasKey) {
            dot.className = "w-2 h-2 rounded-full bg-emerald-400";
            text.textContent = `${names[data.active_provider] || data.active_provider} AI Active`;
            badge.className = "hidden lg:flex items-center gap-1.5 whitespace-nowrap text-xs px-2.5 py-1.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
            if (banner) banner.classList.add("hidden");
        } else {
            dot.className = "w-2 h-2 rounded-full bg-amber-400 animate-pulse";
            text.textContent = "AI Key Missing";
            badge.className = "hidden lg:flex items-center gap-1.5 whitespace-nowrap text-xs px-2.5 py-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20";
            if (banner) banner.classList.remove("hidden");
        }
    } catch (e) {
        console.warn("Could not check settings:", e);
    }
}

function initUI() {
    lucide.createIcons();
    marked.setOptions({
        gfm: true,
        breaks: true,
    });
}

function initVoiceIntegration() {
    const vc = window.voiceController;
    const micBtn = document.getElementById("micBtn");
    const voiceVisualizer = document.getElementById("voiceVisualizer");
    const speechInterim = document.getElementById("speechInterimIndicator");
    const userInput = document.getElementById("userInput");

    // Listening state changes
    vc.onListeningChange = (isListening) => {
        if (isListening) {
            micBtn.classList.add("mic-active");
            voiceVisualizer.classList.remove("hidden");
            speechInterim.classList.remove("hidden");
            document.getElementById("voiceStatusText").textContent = vc.autoDetect
                ? "Listening... speak in any language. I'll stop when you pause."
                : "Listening to your voice... Speak now!";
        } else {
            micBtn.classList.remove("mic-active");
            voiceVisualizer.classList.add("hidden");
            speechInterim.classList.add("hidden");
        }
    };

    // Interim speech (preview as user talks)
    vc.onSpeechInterim = (interimText) => {
        userInput.value = interimText;
    };

    // Final speech result -> Automatically send question (in the detected language, in Auto-detect mode)
    vc.onSpeechResult = (finalText, detectedLanguage) => {
        userInput.value = finalText;
        if (finalText.trim()) {
            submitQuestion(finalText.trim(), detectedLanguage);
            userInput.value = "";
        }
    };

    // Recording sent for transcription + language detection
    vc.onTranscribing = (busy) => {
        voiceVisualizer.classList.toggle("hidden", !busy);
        if (busy) document.getElementById("voiceStatusText").textContent = "Recognising what you said and which language...";
    };

    vc.onVoiceError = (message) => {
        voiceVisualizer.classList.remove("hidden");
        document.getElementById("voiceStatusText").textContent = message;
        setTimeout(() => {
            if (!vc.isListening) voiceVisualizer.classList.add("hidden");
        }, 4000);
    };

    // Mic button click handler
    micBtn.addEventListener("click", () => {
        vc.toggleListening();
    });

    // Auto Voice Readout Toggle
    const toggleBtn = document.getElementById("voiceAutoSpeakToggle");
    const toggleText = document.getElementById("voiceToggleText");
    const toggleIcon = document.getElementById("voiceToggleIcon");

    toggleBtn.addEventListener("click", () => {
        vc.autoSpeak = !vc.autoSpeak;
        if (vc.autoSpeak) {
            toggleText.textContent = "Voice Output: ON";
            toggleIcon.className = "w-4 h-4 text-teal-400";
            toggleBtn.classList.add("border-teal-500/50");
        } else {
            vc.stopSpeaking();
            toggleText.textContent = "Voice Output: OFF";
            toggleIcon.className = "w-4 h-4 text-slate-500";
            toggleBtn.classList.remove("border-teal-500/50");
        }
    });
}

function setupEventListeners() {
    // Colleges drawer (phones)
    document.getElementById("openSidebarBtn").addEventListener("click", openSidebar);
    document.getElementById("closeSidebarBtn").addEventListener("click", closeSidebar);
    document.getElementById("sidebarBackdrop").addEventListener("click", closeSidebar);

    // College cards click in sidebar
    document.querySelectorAll(".college-card").forEach((card) => {
        card.addEventListener("click", () => {
            const cid = parseInt(card.getAttribute("data-college-id"));
            selectCollege(cid);
        });
    });

    // Chat form submit
    const chatForm = document.getElementById("chatForm");
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const input = document.getElementById("userInput");
        const query = input.value.trim();
        if (query) {
            submitQuestion(query);
            input.value = "";
        }
    });

    // Clear Chat
    document.getElementById("clearChatBtn").addEventListener("click", () => {
        if (!activeCollege) return;
        activeConversationId = "conv-" + activeCollegeId + "-" + Date.now();
        renderCollegeBanner(activeCollege);
    });

    // Modals
    setupModals();
}

function setupModals() {
    const scrapeModal = document.getElementById("scrapeModal");
    const openScrapeBtn = document.getElementById("openScrapeModalBtn");
    const closeScrapeBtn = document.getElementById("closeScrapeModal");
    const cancelScrapeBtn = document.getElementById("cancelScrapeBtn");
    const scrapeForm = document.getElementById("scrapeForm");

    openScrapeBtn.addEventListener("click", () => {
        scrapeModal.classList.remove("hidden");
        document.getElementById("scrapeUrlInput").focus();
    });

    const hideScrapeModal = () => {
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
        scrapeModal.classList.add("hidden");
    };

    closeScrapeBtn.addEventListener("click", hideScrapeModal);
    cancelScrapeBtn.addEventListener("click", hideScrapeModal);

    // Scrape form submit
    scrapeForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const url = document.getElementById("scrapeUrlInput").value.trim();
        const maxPages = parseInt(document.getElementById("scrapeMaxPages").value) || 120;
        const maxDepth = parseInt(document.getElementById("scrapeMaxDepth").value) || 4;

        const progressBox = document.getElementById("crawlProgressBox");
        const progressBar = document.getElementById("crawlProgressBar");
        const statusLabel = document.getElementById("crawlStatusLabel");
        const countBadge = document.getElementById("crawlCountBadge");
        const currentUrlText = document.getElementById("crawlCurrentUrl");
        const startBtn = document.getElementById("startScrapeBtn");

        progressBox.classList.remove("hidden");
        startBtn.disabled = true;
        startBtn.innerHTML = `<span class="animate-spin inline-block mr-1">⌛</span> Crawling...`;

        try {
            const resp = await fetch("/api/crawl", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url, max_pages: maxPages, max_depth: maxDepth })
            });

            if (!resp.ok) {
                const errText = await resp.text();
                let errMsg = errText;
                try {
                    const parsed = JSON.parse(errText);
                    if (parsed.detail) errMsg = parsed.detail;
                } catch(e) {}
                alert(errMsg);
                startBtn.disabled = false;
                startBtn.innerHTML = `Start Crawling & Indexing`;
                return;
            }

            const data = await resp.json();

            if (data.crawl_id) {
                listenToCrawlProgress(data.crawl_id, data.max_pages || maxPages);
            }
        } catch (err) {
            alert("Failed to start crawler: " + err.message);
            startBtn.disabled = false;
            startBtn.innerHTML = `Start Crawling & Indexing`;
        }
    });

    // Settings Modal
    const settingsModal = document.getElementById("settingsModal");
    const openSettingsBtn = document.getElementById("openSettingsBtn");
    const aiStatusBtn = document.getElementById("aiStatusBtn");
    const closeSettingsBtn = document.getElementById("closeSettingsModal");
    const settingsForm = document.getElementById("settingsForm");

    const openSettings = async () => {
        try {
            const res = await fetch("/api/settings");
            const data = await res.json();
            if (data.gemini_masked_key) {
                document.getElementById("geminiKeyInput").placeholder = data.gemini_masked_key;
            }
            if (data.openai_masked_key) {
                document.getElementById("openaiKeyInput").placeholder = data.openai_masked_key;
            }
            if (data.groq_masked_key) {
                document.getElementById("groqKeyInput").placeholder = `${data.groq_masked_key} (saved)`;
            }
            if (data.openrouter_masked_key) {
                document.getElementById("openrouterKeyInput").placeholder = `${data.openrouter_masked_key} (saved)`;
            }
            document.getElementById("openrouterModelInput").value = data.openrouter_model || "";
            document.getElementById("providerSelect").value = data.default_provider || "gemini";
            document.getElementById("settingsStatus").classList.add("hidden");
        } catch (e) {
            console.error("Failed to load settings:", e);
        }
        settingsModal.classList.remove("hidden");
    };

    openSettingsBtn.addEventListener("click", openSettings);
    if (aiStatusBtn) aiStatusBtn.addEventListener("click", openSettings);

    closeSettingsBtn.addEventListener("click", () => settingsModal.classList.add("hidden"));

    const showSettingsStatus = (ok, message) => {
        const box = document.getElementById("settingsStatus");
        box.className = ok
            ? "text-xs rounded-lg px-3 py-2 border border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
            : "text-xs rounded-lg px-3 py-2 border border-amber-500/40 bg-amber-500/10 text-amber-300";
        box.textContent = message;
    };

    settingsForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const keyFields = {
            gemini_api_key: "geminiKeyInput",
            openai_api_key: "openaiKeyInput",
            groq_api_key: "groqKeyInput",
            openrouter_api_key: "openrouterKeyInput",
        };
        const payload = { default_provider: document.getElementById("providerSelect").value };
        for (const [field, inputId] of Object.entries(keyFields)) {
            const value = document.getElementById(inputId).value.trim();
            if (value) payload[field] = value;   // empty box = keep the saved key
        }
        const model = document.getElementById("openrouterModelInput").value.trim();
        if (model) payload.openrouter_model = model;

        window.voiceController.preferredVoice = document.getElementById("voiceSelect").value;

        const saveBtn = document.getElementById("saveSettingsBtn");
        saveBtn.disabled = true;
        saveBtn.textContent = "Saving & testing key...";
        showSettingsStatus(true, "Saving, then sending a tiny test question to check the key works...");
        try {
            const resp = await fetch("/api/settings", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (!resp.ok) {
                showSettingsStatus(false, `Not saved: ${data.detail || resp.statusText}`);
                return;
            }
            let message = data.check.ok ? `✓ Saved. ${data.check.message}` : `Saved, but: ${data.check.message}`;
            if (!data.saved_to_env) message += " (Could not write the .env file, so this lasts until the server restarts.)";
            showSettingsStatus(data.check.ok, message);
            // Clear typed keys; the placeholders show the saved (masked) keys instead
            Object.values(keyFields).forEach((id) => { document.getElementById(id).value = ""; });
            await checkAISettings();
            const fresh = await (await fetch("/api/settings")).json();
            if (fresh.groq_masked_key) document.getElementById("groqKeyInput").placeholder = `${fresh.groq_masked_key} (saved)`;
            if (fresh.openrouter_masked_key) document.getElementById("openrouterKeyInput").placeholder = `${fresh.openrouter_masked_key} (saved)`;
            if (fresh.gemini_masked_key) document.getElementById("geminiKeyInput").placeholder = `${fresh.gemini_masked_key} (saved)`;
            if (fresh.openai_masked_key) document.getElementById("openaiKeyInput").placeholder = `${fresh.openai_masked_key} (saved)`;
            if (data.check.ok) setTimeout(() => settingsModal.classList.add("hidden"), 2500);
        } catch (err) {
            showSettingsStatus(false, `Couldn't reach the server: ${err.message}`);
        } finally {
            saveBtn.disabled = false;
            saveBtn.textContent = "Save Settings";
        }
    });

    // Language Selector Change
    const langSelect = document.getElementById("languageSelect");
    if (langSelect) {
        langSelect.addEventListener("change", (e) => {
            applyLanguage(e.target.value);
        });
    }
}

function isAutoLanguage() {
    return currentLanguage() === "auto";
}

async function applyLanguage(langCode) {
    if (!appLanguages || Object.keys(appLanguages).length === 0) {
        await loadLanguages();
    }
    const vc = window.voiceController;
    vc.autoDetect = langCode === "auto";
    if (vc.autoDetect) {
        // The language is worked out per question: from the voice recording, or from the typed text's script
        const input = document.getElementById("userInput");
        if (input) input.placeholder = "Ask in any language, by typing or tapping the mic...";
        renderSuggestions();
        return;
    }

    const info = appLanguages[langCode] || appLanguages["en"];
    if (!info) return;

    // Update voice controller
    window.voiceController.setLanguage(langCode, info.speech_locale, info.default_voice);

    // Update settings voice dropdown selection
    const voiceSelect = document.getElementById("voiceSelect");
    if (voiceSelect && info.default_voice) {
        voiceSelect.value = info.default_voice;
    }

    // Update input box placeholder
    const input = document.getElementById("userInput");
    if (input && info.input_placeholder) {
        input.placeholder = info.input_placeholder;
    }

    // Example questions follow the chosen language
    renderSuggestions();
}

function listenToCrawlProgress(crawlId, maxPages) {
    const progressBar = document.getElementById("crawlProgressBar");
    const statusLabel = document.getElementById("crawlStatusLabel");
    const countBadge = document.getElementById("crawlCountBadge");
    const currentUrlText = document.getElementById("crawlCurrentUrl");
    const startBtn = document.getElementById("startScrapeBtn");

    if (currentEventSource) currentEventSource.close();
    currentEventSource = new EventSource(`/api/crawl/stream/${crawlId}`);

    currentEventSource.onmessage = (event) => {
        if (!event.data) return;
        const msg = JSON.parse(event.data);

        if (msg.type === "status" && msg.data.message) {
            statusLabel.textContent = msg.data.message;
        } else if (msg.type === "page_start") {
            currentUrlText.textContent = `Scraping: ${msg.data.url}`;
            statusLabel.textContent = `Scraping page ${msg.data.scraped_count + 1} of ${maxPages}...`;
        } else if (msg.type === "page_success") {
            const pct = Math.min(100, Math.round((msg.data.scraped_count / maxPages) * 100));
            progressBar.style.width = `${pct}%`;
            countBadge.textContent = `${msg.data.scraped_count} / ${maxPages}`;
            statusLabel.textContent = `Indexed "${msg.data.title.substring(0, 30)}..." (${msg.data.chunks} chunks)`;
        } else if (msg.type === "completed" || msg.type === "done") {
            progressBar.style.width = "100%";
            statusLabel.textContent = msg.data.warning || "Crawl and Vector Indexing Complete!";
            currentEventSource.close();
            currentEventSource = null;

            setTimeout(async () => {
                document.getElementById("scrapeModal").classList.add("hidden");
                startBtn.disabled = false;
                startBtn.innerHTML = `Start Crawling & Indexing`;
                document.getElementById("crawlProgressBox").classList.add("hidden");

                await reloadCollegesList();
                const newCollegeId = msg.data.college_id;
                if (newCollegeId) selectCollege(newCollegeId);
            }, msg.data.warning ? 6000 : 1200);
        } else if (msg.type === "error") {
            statusLabel.textContent = `Error: ${msg.data.error}`;
            currentEventSource.close();
            startBtn.disabled = false;
            startBtn.innerHTML = `Start Crawling & Indexing`;
        }
    };

    currentEventSource.onerror = () => {
        if (currentEventSource) {
            currentEventSource.close();
            currentEventSource = null;
        }
    };
}

async function reloadCollegesList() {
    try {
        const resp = await fetch("/api/colleges");
        const colleges = await resp.json();

        // Update count badge
        document.getElementById("collegeCountBadge").textContent = colleges.length;

        // Update list sidebar
        const list = document.getElementById("collegesList");
        if (colleges.length === 0) {
            list.innerHTML = `
                <div class="text-center py-12 px-4">
                    <p class="text-sm font-medium text-slate-300">No colleges indexed yet</p>
                    <p class="text-xs text-slate-500 mt-1">Click "Add College" to begin.</p>
                </div>
            `;
            return;
        }

        list.innerHTML = "";
        colleges.forEach((col) => {
            const card = document.createElement("div");
            card.className = `college-card cursor-pointer p-3 rounded-xl border border-slate-700/50 hover:border-teal-500/50 bg-slate-800/80 hover:bg-slate-750 transition-all flex flex-col gap-1.5 group ${activeCollegeId === col.id ? 'active' : ''}`;
            card.setAttribute("data-college-id", col.id);
            card.innerHTML = `
                <div class="flex items-start justify-between">
                    <h3 class="font-medium text-sm text-slate-200 group-hover:text-teal-300 truncate max-w-[180px]">${escapeHtml(col.name || col.domain)}</h3>
                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400 font-mono">${col.page_count} pgs</span>
                </div>
                <p class="text-xs text-slate-400 truncate">${col.url}</p>
                <div class="flex items-center justify-between text-[11px] text-slate-500 mt-1">
                    <span>${col.chunk_count} chunks</span>
                    <button onclick="deleteCollege(${col.id}, event)" class="md:opacity-0 md:group-hover:opacity-100 hover:text-red-400 transition" title="Delete College">
                        <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                    </button>
                </div>
            `;
            card.addEventListener("click", () => selectCollege(col.id));
            list.appendChild(card);
        });
        lucide.createIcons();
    } catch (e) {
        console.error("Error refreshing colleges:", e);
    }
}

// Subtitle under the college name; while facts are still being extracted it shows progress and keeps polling
function updateCollegeSubtitle(col) {
    const subtitle = document.getElementById("currentCollegeSubtitle");
    let text = `${col.url} • ${col.page_count} pages • ${col.fact_count || 0} facts`;
    if (col.extraction) {
        const { done, total } = col.extraction;
        text += total ? ` • building course & company lists (${done}/${total} pages)…` : " • building course & company lists…";
    }
    subtitle.textContent = text;

    const banner = document.getElementById("extractionNotice");
    if (banner) banner.classList.toggle("hidden", !col.extraction);

    clearTimeout(extractionPollTimer);
    if (col.extraction) {
        extractionPollTimer = setTimeout(async () => {
            if (activeCollegeId !== col.id) return;
            try {
                const resp = await fetch(`/api/colleges/${col.id}`);
                const fresh = await resp.json();
                if (activeCollegeId === fresh.id) {
                    activeCollege = fresh;
                    updateCollegeSubtitle(fresh);
                }
            } catch (e) { /* try again on next selection */ }
        }, 10000);
    }
}

function renderCollegeBanner(col) {
    const chatMessages = document.getElementById("chatMessages");
    chatMessages.innerHTML = `
        <div class="text-center py-5 px-4 bg-slate-800/40 rounded-2xl border border-slate-700/40 mb-4 max-w-2xl mx-auto">
            <span class="text-xs px-2.5 py-1 rounded-full bg-teal-500/10 text-teal-300 font-medium border border-teal-500/20">Ready to answer</span>
            <h3 class="text-base font-bold text-slate-100 mt-2">${escapeHtml(col.name || col.domain)}</h3>
            <p class="text-xs text-slate-400 mt-1">${col.page_count} pages read. Ask anything, or tap a question:</p>
            <p id="extractionNotice" class="${col.extraction ? "" : "hidden"} text-xs text-amber-300 mt-2">
                Still building the complete course &amp; company lists. "List all…" answers will be partial until this finishes.
            </p>
            <div class="suggestion-row mt-3 flex flex-wrap justify-center gap-2"></div>
        </div>
    `;
    renderSuggestions();
}

async function selectCollege(collegeId) {
    activeCollegeId = collegeId;
    activeConversationId = "conv-" + collegeId + "-" + Date.now();
    try { localStorage.setItem("lastCollegeId", String(collegeId)); } catch (e) { /* storage unavailable */ }
    closeSidebar();

    // Highlight card
    document.querySelectorAll(".college-card").forEach((c) => {
        c.classList.remove("active");
        if (parseInt(c.getAttribute("data-college-id")) === collegeId) {
            c.classList.add("active");
        }
    });

    // Fetch college details
    try {
        const resp = await fetch(`/api/colleges/${collegeId}`);
        const col = await resp.json();
        activeCollege = col;

        document.getElementById("currentCollegeTitle").textContent = col.name || col.domain;
        updateCollegeSubtitle(col);

        // Fetch pages preview
        const pagesResp = await fetch(`/api/colleges/${collegeId}/pages?limit=6`);
        const pages = await pagesResp.json();
        const previewList = document.getElementById("pagesPreviewList");
        previewList.innerHTML = "";
        pages.forEach((p) => {
            previewList.innerHTML += `<div class="truncate hover:text-teal-300"><a href="${p.url}" target="_blank" rel="noopener">📄 ${escapeHtml(p.title || p.url)}</a></div>`;
        });
        document.getElementById("collegeDetailBox").classList.remove("hidden");

        // Clear previous messages and show college banner with example questions
        renderCollegeBanner(col);
    } catch (err) {
        console.error("Error selecting college:", err);
    }
}

async function deleteCollege(collegeId, event) {
    event.stopPropagation();
    if (!confirm("Are you sure you want to delete this college and its indexed data?")) return;

    try {
        await fetch(`/api/colleges/${collegeId}`, { method: "DELETE" });
        if (activeCollegeId === collegeId) {
            activeCollegeId = null;
            activeCollege = null;
            try { localStorage.removeItem("lastCollegeId"); } catch (e) { /* storage unavailable */ }
            document.getElementById("currentCollegeTitle").textContent = "Choose or add a college to begin";
            document.getElementById("currentCollegeSubtitle").textContent = "Ask any question by voice or text.";
            document.getElementById("collegeDetailBox").classList.add("hidden");
        }
        await reloadCollegesList();
    } catch (e) {
        alert("Delete failed: " + e.message);
    }
}

function fillPrompt(text) {
    document.getElementById("userInput").value = text;
    document.getElementById("userInput").focus();
}

async function submitQuestion(query, detectedLanguage = null) {
    if (!activeCollegeId) {
        alert("Please choose a college first, or add one.");
        if (window.innerWidth < 768) openSidebar();
        return;
    }

    const chatMessages = document.getElementById("chatMessages");
    const sendBtn = document.getElementById("sendBtn");
    const voiceController = window.voiceController;

    // 1. Render User Message
    appendUserMessage(query);

    // 2. Render Loading Assistant Placeholder
    const loadingId = "loading-" + Date.now();
    const loadingDiv = document.createElement("div");
    loadingDiv.id = loadingId;
    loadingDiv.className = "flex items-start space-x-3 max-w-3xl";
    loadingDiv.innerHTML = `
        <div class="w-8 h-8 rounded-lg bg-teal-500/20 text-teal-400 flex items-center justify-center flex-shrink-0 mt-0.5">
            <i data-lucide="bot" class="w-4 h-4 animate-spin"></i>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/60 rounded-2xl rounded-tl-sm p-4 text-xs text-slate-400 flex items-center gap-2">
            <span>Searching college knowledge base & formulating answer...</span>
        </div>
    `;
    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    lucide.createIcons();

    sendBtn.disabled = true;

    try {
        // Auto-detect: a spoken question carries the language Whisper heard; a typed one is detected by the server
        const autoMode = isAutoLanguage();
        const currentLang = detectedLanguage || currentLanguage();

        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                college_id: activeCollegeId,
                query: query,
                conversation_id: activeConversationId,
                voice: autoMode ? null : voiceController.preferredVoice,
                language: currentLang,
                generate_audio: true
            })
        });

        const data = await resp.json();
        loadingDiv.remove();

        if (data.error) {
            appendAssistantMessage("⚠️ " + data.error);
            return;
        }

        // If backend returned a specific language / default voice, sync it
        if (data.default_voice && !autoMode) {
            voiceController.preferredVoice = data.default_voice;
        }

        // Render Assistant Response with citations and voice controls
        const languageNote = autoMode && data.language_name ? `Answered in ${data.language_name} (detected automatically)` : "";
        appendAssistantMessage(data.content, data.spoken_text, data.citations, data.audio_url, data.speech_locale, languageNote);

        // Voice output with regional speech locale
        if (voiceController.autoSpeak && (data.audio_url || data.spoken_text)) {
            voiceController.speak(data.spoken_text, data.audio_url, data.speech_locale);
        }

    } catch (err) {
        loadingDiv.remove();
        appendAssistantMessage("⚠️ Failed to get answer: " + err.message);
    } finally {
        sendBtn.disabled = false;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function appendUserMessage(text) {
    const chatMessages = document.getElementById("chatMessages");
    const div = document.createElement("div");
    div.className = "flex justify-end mb-3";
    div.innerHTML = `
        <div class="bg-gradient-to-r from-teal-600 to-emerald-700 text-white rounded-2xl rounded-tr-sm px-4 py-3 max-w-2xl text-sm shadow-md">
            <p>${escapeHtml(text)}</p>
        </div>
    `;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function appendAssistantMessage(content, spokenText, citations = [], audioUrl = null, speechLocale = null, languageNote = "") {
    const chatMessages = document.getElementById("chatMessages");
    const msgId = "msg-" + Date.now();
    const div = document.createElement("div");
    div.id = msgId;
    div.className = "flex items-start gap-2 sm:gap-3 max-w-3xl mb-4";

    // Format citations pills
    let citationsHtml = "";
    if (citations && citations.length > 0) {
        citationsHtml = `
            <div class="mt-3 pt-3 border-t border-slate-700/60">
                <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5 flex items-center gap-1">
                    <i data-lucide="link" class="w-3 h-3 text-teal-400"></i> Sources & Citations
                </span>
                <div class="flex flex-wrap gap-1.5">
                    ${citations.map((c) => `
                        <a href="${c.url}" target="_blank" rel="noopener" class="inline-flex items-center gap-1 text-[11px] px-2.5 py-1 rounded-md bg-slate-750 hover:bg-slate-700 text-slate-300 hover:text-teal-300 border border-slate-700/80 transition">
                            <span>📄 ${escapeHtml(c.title || c.url)}</span>
                            <i data-lucide="external-link" class="w-3 h-3 text-slate-500"></i>
                        </a>
                    `).join("")}
                </div>
            </div>
        `;
    }

    // Wrap tables so wide ones scroll sideways instead of stretching the chat
    const htmlContent = marked.parse(content || "")
        .replace(/<table>/g, '<div class="table-wrap"><table>')
        .replace(/<\/table>/g, "</table></div>");

    div.innerHTML = `
        <div class="hidden sm:flex w-8 h-8 rounded-lg bg-teal-500/20 text-teal-400 items-center justify-center flex-shrink-0 mt-0.5 shadow-sm">
            <i data-lucide="bot" class="w-4 h-4"></i>
        </div>
        <div class="bg-slate-800/80 border border-slate-700/60 rounded-2xl rounded-tl-sm p-3 sm:p-4 text-slate-200 shadow-md flex-1 min-w-0">
            <div class="prose-chat">${htmlContent}</div>
            ${citationsHtml}
            
            <!-- Voice action bar for this message -->
            <div class="mt-3 pt-2 flex items-center justify-between text-xs text-slate-400">
                <div class="flex items-center space-x-2">
                    <button class="speak-btn flex items-center gap-1 px-2.5 py-1 rounded bg-slate-750 hover:bg-slate-700 text-teal-300 transition" title="Speak Answer Aloud">
                        <i data-lucide="volume-2" class="w-3.5 h-3.5"></i>
                        <span>Listen Voice</span>
                    </button>
                    ${audioUrl ? `<span class="text-[10px] text-slate-500 font-mono">Neural Audio</span>` : ''}
                </div>
                ${languageNote ? `<span class="text-[11px] text-teal-300/80 flex items-center gap-1"><i data-lucide="languages" class="w-3.5 h-3.5"></i>${escapeHtml(languageNote)}</span>` : ''}
            </div>
        </div>
    `;

    // Bind speak button
    const speakBtn = div.querySelector(".speak-btn");
    speakBtn.addEventListener("click", () => {
        window.voiceController.speak(spokenText || content, audioUrl, speechLocale);
    });

    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    lucide.createIcons();
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
