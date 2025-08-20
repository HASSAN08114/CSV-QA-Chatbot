def apply_custom_css():
    css = """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            /* CSS Variables for Theming */
            :root {
                /* Light Theme Colors */
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-tertiary: #f3f4f6;
                --text-primary: #111827;
                --text-secondary: #374151;
                --text-muted: #6b7280;
                --border-color: #e5e7eb;
                --border-light: #d1d5db;
                --accent-primary: #3b82f6;
                --accent-hover: #2563eb;
                --success-bg: #ecfdf5;
                --success-border: #a7f3d0;
                --success-text: #065f46;
                --warning-bg: #fffbeb;
                --warning-border: #fcd34d;
                --warning-text: #92400e;
                --info-bg: #eff6ff;
                --info-border: #93c5fd;
                --info-text: #1e40af;
                --shadow-light: 0 1px 2px rgba(0, 0, 0, 0.05);
                --shadow-medium: 0 2px 8px rgba(0, 0, 0, 0.15);
            }
            
            /* Dark Theme Colors */
            [data-theme="dark"] {
                --bg-primary: #111827;
                --bg-secondary: #1f2937;
                --bg-tertiary: #374151;
                --text-primary: #f9fafb;
                --text-secondary: #d1d5db;
                --text-muted: #9ca3af;
                --border-color: #374151;
                --border-light: #4b5563;
                --accent-primary: #3b82f6;
                --accent-hover: #60a5fa;
                --success-bg: #064e3b;
                --success-border: #059669;
                --success-text: #a7f3d0;
                --warning-bg: #78350f;
                --warning-border: #d97706;
                --warning-text: #fcd34d;
                --info-bg: #1e3a8a;
                --info-border: #3b82f6;
                --info-text: #93c5fd;
                --shadow-light: 0 1px 2px rgba(0, 0, 0, 0.3);
                --shadow-medium: 0 2px 8px rgba(0, 0, 0, 0.4);
            }
            
            /* Reset and Global Styles */
            * {
                box-sizing: border-box;
            }
            
            html, body, [class^="st-"] {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                background: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                line-height: 1.6 !important;
                transition: background-color 0.3s ease, color 0.3s ease !important;
            }
            
            /* Theme Toggle Button */
            .theme-toggle {
                position: fixed !important;
                top: 1rem !important;
                right: 1rem !important;
                z-index: 1002 !important;
                background: var(--accent-primary) !important;
                color: white !important;
                border: none !important;
                border-radius: 50% !important;
                width: 40px !important;
                height: 40px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                cursor: pointer !important;
                box-shadow: var(--shadow-medium) !important;
                transition: all 0.2s ease !important;
                font-size: 1.2rem !important;
            }
            
            .theme-toggle:hover {
                background: var(--accent-hover) !important;
                transform: scale(1.05) !important;
            }
            
            /* Remove text selection highlighting globally but allow in chat */
            ::selection {
                background: rgba(59, 130, 246, 0.2) !important;
                color: inherit !important;
            }
            
            /* Fix chat input styling */
            .stChatInput {
                border: 2px solid var(--border-color) !important;
                border-radius: 12px !important;
                background: var(--bg-primary) !important;
                box-shadow: var(--shadow-light) !important;
                transition: all 0.2s ease !important;
            }
            
            .stChatInput:focus {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            }
            
            .stChatInput:hover {
                border-color: var(--accent-hover) !important;
            }
            
            ::-moz-selection {
                background: rgba(59, 130, 246, 0.2) !important;
                color: inherit !important;
            }
            
            /* Disable text selection for UI elements but allow in chat messages */
            .sidebar-card, .main-header, .stButton, .stTextInput, .stTextArea, .stTabs {
                -webkit-user-select: none !important;
                -moz-user-select: none !important;
                -ms-user-select: none !important;
                user-select: none !important;
            }
            
            /* Allow text selection in chat messages */
            .chat-message, .message-content {
                -webkit-user-select: text !important;
                -moz-user-select: text !important;
                -ms-user-select: text !important;
                user-select: text !important;
                cursor: text !important;
            }
            
            /* Better selection styling for chat messages */
            .chat-message ::selection,
            .message-content ::selection {
                background: rgba(59, 130, 246, 0.3) !important;
                color: var(--text-primary) !important;
            }
            
            .chat-message ::-moz-selection,
            .message-content ::-moz-selection {
                background: rgba(59, 130, 246, 0.3) !important;
                color: var(--text-primary) !important;
            }
            
            /* Main Container - ChatGPT Style */
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
                margin-left: 280px !important;
                background: var(--bg-primary) !important;
                min-height: 100vh !important;
                transition: background-color 0.3s ease !important;
            }
            
            /* Header - Clean and Minimal */
            .main-header {
                background: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                padding: 1.5rem 2rem 1rem 2rem !important;
                margin: 0 !important;
                border-bottom: 1px solid var(--border-color) !important;
                text-align: center !important;
                transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important;
            }
            
            .main-header h1 {
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                margin: 0 0 0.25rem 0 !important;
                color: var(--text-primary) !important;
                transition: color 0.3s ease !important;
            }
            
            .main-header p {
                font-size: 0.875rem !important;
                color: var(--text-muted) !important;
                margin: 0 !important;
                font-weight: 400 !important;
                transition: color 0.3s ease !important;
            }
            
            /* Sidebar Toggle Button */
            .sidebar-toggle {
                position: fixed !important;
                top: 1rem !important;
                left: 1rem !important;
                z-index: 1001 !important;
                background: var(--accent-primary) !important;
                color: white !important;
                border: none !important;
                border-radius: 50% !important;
                width: 40px !important;
                height: 40px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                cursor: pointer !important;
                box-shadow: var(--shadow-medium) !important;
                transition: all 0.2s ease !important;
            }
            
            .sidebar-toggle:hover {
                background: var(--accent-hover) !important;
                transform: scale(1.05) !important;
            }
            
            /* Sidebar - Clean and Minimal */
            .css-1d391kg {
                background: var(--bg-secondary) !important;
                border-right: 1px solid var(--border-color) !important;
                padding: 1rem !important;
                width: 280px !important;
                min-width: 280px !important;
                max-width: 280px !important;
                overflow-y: auto !important;
                overflow-x: hidden !important;
                position: fixed !important;
                height: 100vh !important;
                top: 0 !important;
                left: 0 !important;
                z-index: 1000 !important;
                box-sizing: border-box !important;
                transform: translateX(0) !important;
                transition: transform 0.3s ease, background-color 0.3s ease, border-color 0.3s ease !important;
            }
            
            /* Hide all default Streamlit sidebar content except our custom elements */
            .css-1d391kg > *:not(.sidebar-header):not(.sidebar-card) {
                display: none !important;
            }
            
            /* Ensure only our custom sidebar content is visible */
            .css-1d391kg .sidebar-header,
            .css-1d391kg .sidebar-card {
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
            
            /* Hide any remaining Streamlit elements */
            .css-1d391kg [data-testid],
            .css-1d391kg [class*="st-"]:not(.sidebar-header):not(.sidebar-card) {
                display: none !important;
            }
            
            /* Sidebar collapsed state */
            .sidebar-collapsed {
                transform: translateX(-100%) !important;
            }
            
            /* Main content expanded when sidebar collapsed */
            .content-expanded {
                margin-left: 0 !important;
            }
            
            /* Hide Streamlit's default sidebar elements */
            .css-1d391kg > div {
                width: 100% !important;
                max-width: 100% !important;
                overflow: hidden !important;
            }
            
            /* Sidebar Header */
            .sidebar-header {
                background: var(--bg-primary) !important;
                padding: 1rem !important;
                margin: -1rem -1rem 1rem -1rem !important;
                border-bottom: 1px solid var(--border-color) !important;
                text-align: center !important;
                transition: background-color 0.3s ease, border-color 0.3s ease !important;
            }
            
            .sidebar-header h2 {
                font-size: 1rem !important;
                font-weight: 600 !important;
                color: var(--text-primary) !important;
                margin: 0 !important;
                transition: color 0.3s ease !important;
            }
            
            .css-1d391kg .stRadio > label {
                background: var(--bg-primary) !important;
                border: 1px solid var(--border-light) !important;
                border-radius: 6px !important;
                padding: 0.5rem 0.75rem !important;
                margin: 0.125rem 0 !important;
                font-weight: 500 !important;
                font-size: 0.875rem !important;
                transition: all 0.15s ease !important;
                color: var(--text-secondary) !important;
            }
            
            .css-1d391kg .stRadio > label:hover {
                border-color: var(--accent-primary) !important;
                background: var(--info-bg) !important;
            }
            
            .css-1d391kg .stRadio > label[data-baseweb="radio"] {
                background: var(--accent-primary) !important;
                color: white !important;
                border-color: var(--accent-primary) !important;
            }
            
            /* Sidebar Cards - Minimal */
            .sidebar-card {
                background: var(--bg-primary) !important;
                border-radius: 8px !important;
                padding: 1rem !important;
                margin: 0.75rem 0 !important;
                border: 1px solid var(--border-color) !important;
                box-shadow: var(--shadow-light) !important;
                width: 100% !important;
                box-sizing: border-box !important;
                overflow: hidden !important;
                transition: background-color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease !important;
            }
            
            .sidebar-card h3 {
                font-size: 0.875rem !important;
                font-weight: 600 !important;
                color: var(--text-secondary) !important;
                margin: 0 0 0.75rem 0 !important;
                transition: color 0.3s ease !important;
            }
            
            /* Toggle indicators */
            .toggle-indicator {
                display: inline-flex !important;
                align-items: center !important;
                gap: 0.5rem !important;
                padding: 0.5rem 0.75rem !important;
                border-radius: 6px !important;
                margin: 0.25rem 0 !important;
                transition: all 0.15s ease !important;
            }
            
            .toggle-active {
                background: var(--success-bg) !important;
                border: 1px solid var(--success-border) !important;
                color: var(--success-text) !important;
            }
            
            .toggle-inactive {
                background: var(--warning-bg) !important;
                border: 1px solid var(--warning-border) !important;
                color: var(--warning-text) !important;
            }
            
            /* Hide Streamlit's default icons */
            .css-1d391kg [data-testid="stSidebarNav"] {
                display: none !important;
            }
            
            /* Fix any horizontal scrolling */
            .css-1d391kg * {
                max-width: 100% !important;
                box-sizing: border-box !important;
            }
            
            /* Ensure sidebar content doesn't overflow */
            .css-1d391kg .stRadio > div {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            .css-1d391kg .stTextInput > div {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            .css-1d391kg .stSelectbox > div {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            /* Hide any default Streamlit navigation */
            [data-testid="stSidebarNav"] {
                display: none !important;
            }
            
            /* Hide specific problematic elements */
            .css-1d391kg [data-testid="stSidebarNav"],
            .css-1d391kg [data-testid="stSidebarNavItems"],
            .css-1d391kg [data-testid="stSidebarNavLink"] {
                display: none !important;
            }
            
            /* Hide navigation elements and any elements with navigation-related classes */
            .css-1d391kg [class*="nav"],
            .css-1d391kg [class*="menu"],
            .css-1d391kg [class*="sidebar-nav"],
            .css-1d391kg [class*="navigation"] {
                display: none !important;
            }
            
            /* More specific selectors to hide navigation elements */
            .css-1d391kg > div:first-child:not(.sidebar-header) {
                display: none !important;
            }
            
            /* Ensure sidebar header is visible and properly styled */
            .sidebar-header {
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }
            
            /* Hide any elements that might be navigation-related */
            .css-1d391kg > div:not(.sidebar-card):not(.sidebar-header) {
                display: none !important;
            }
            
            /* Target specific Streamlit sidebar elements */
            .css-1d391kg [role="navigation"],
            .css-1d391kg [aria-label*="navigation"],
            .css-1d391kg [aria-label*="menu"] {
                display: none !important;
            }
            
            /* Ensure proper spacing */
            .css-1d391kg .stMarkdown {
                width: 100% !important;
                max-width: 100% !important;
            }
            
            /* Buttons - Clean */
            .stButton > button {
                background: var(--accent-primary) !important;
                color: white !important;
                border: none !important;
                border-radius: 6px !important;
                padding: 0.5rem 1rem !important;
                font-weight: 500 !important;
                font-size: 0.875rem !important;
                transition: all 0.15s ease !important;
            }
            
            .stButton > button:hover {
                background: var(--accent-hover) !important;
            }
            
            /* Input Fields - Clean */
            .stTextInput > div > div > input {
                border: 1px solid var(--border-light) !important;
                border-radius: 6px !important;
                padding: 0.5rem 0.75rem !important;
                font-size: 0.875rem !important;
                background: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                transition: border-color 0.3s ease, background-color 0.3s ease, color 0.3s ease !important;
            }
            
            .stTextInput > div > div > input:focus {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            }
            
            /* Chat Interface - ChatGPT Style */
            .chat-container {
                padding: 0 !important;
                background: var(--bg-primary) !important;
                min-height: calc(100vh - 120px) !important;
                display: flex !important;
                flex-direction: column !important;
                width: 100% !important;
                max-width: 100% !important;
                transition: background-color 0.3s ease !important;
            }
            
            /* Chat Messages - ChatGPT Style */
            .chat-message {
                padding: 1.5rem 2rem !important;
                margin: 0 !important;
                border-bottom: 1px solid var(--border-color) !important;
                position: relative !important;
                transition: background-color 0.3s ease, border-color 0.3s ease !important;
            }
            
            .user-message {
                background: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
            }
            
            .bot-message {
                background: var(--bg-primary) !important;
                color: var(--text-primary) !important;
            }
            
            .message-header {
                display: flex !important;
                align-items: center !important;
                gap: 0.75rem !important;
                margin-bottom: 0.75rem !important;
            }
            
            .message-avatar {
                width: 28px !important;
                height: 28px !important;
                border-radius: 50% !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                font-size: 0.875rem !important;
                font-weight: 600 !important;
            }
            
            .user-message .message-avatar {
                background: var(--accent-primary) !important;
                color: white !important;
            }
            
            .bot-message .message-avatar {
                background: #10b981 !important;
                color: white !important;
            }
            
            .message-role {
                font-weight: 600 !important;
                font-size: 0.875rem !important;
                color: var(--text-secondary) !important;
                transition: color 0.3s ease !important;
            }
            
            .message-content {
                font-size: 0.875rem !important;
                line-height: 1.6 !important;
                color: var(--text-primary) !important;
                margin-left: 2.5rem !important;
                transition: color 0.3s ease !important;
            }
            
            .message-timestamp {
                font-size: 0.75rem !important;
                color: var(--text-muted) !important;
                margin-top: 0.5rem !important;
                margin-left: 2.5rem !important;
                transition: color 0.3s ease !important;
            }
            
            /* Chat Input - ChatGPT Style */
            .stTextArea > div > textarea {
                border: 1px solid var(--border-light) !important;
                border-radius: 12px !important;
                padding: 0.75rem 1rem !important;
                font-size: 0.875rem !important;
                background: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                resize: none !important;
                min-height: 48px !important;
                box-shadow: var(--shadow-light) !important;
                transition: border-color 0.3s ease, background-color 0.3s ease, color 0.3s ease !important;
            }
            
            .stTextArea > div > textarea:focus {
                border-color: var(--accent-primary) !important;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            }
            
            /* Tabs - Clean */
            .stTabs [data-baseweb="tab-list"] {
                background: var(--bg-primary) !important;
                border-radius: 6px !important;
                padding: 0.25rem !important;
                margin: 0.75rem 0 !important;
                border: 1px solid var(--border-color) !important;
                transition: background-color 0.3s ease, border-color 0.3s ease !important;
            }
            
            .stTabs [data-baseweb="tab"] {
                background: transparent !important;
                border: none !important;
                border-radius: 4px !important;
                padding: 0.5rem 1rem !important;
                font-weight: 500 !important;
                font-size: 0.875rem !important;
                color: var(--text-muted) !important;
                transition: all 0.15s ease !important;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background: var(--bg-secondary) !important;
                color: var(--text-secondary) !important;
            }
            
            .stTabs [aria-selected="true"] {
                background: var(--accent-primary) !important;
                color: white !important;
            }
            
            /* Status Messages - Clean */
            .stSuccess {
                background: var(--success-bg) !important;
                border: 1px solid var(--success-border) !important;
                border-radius: 6px !important;
                color: var(--success-text) !important;
                padding: 0.75rem !important;
                font-size: 0.875rem !important;
            }
            
            .stWarning {
                background: var(--warning-bg) !important;
                border: 1px solid var(--warning-border) !important;
                border-radius: 6px !important;
                color: var(--warning-text) !important;
                padding: 0.75rem !important;
                font-size: 0.875rem !important;
            }
            
            .stInfo {
                background: var(--info-bg) !important;
                border: 1px solid var(--info-border) !important;
                border-radius: 6px !important;
                color: var(--info-text) !important;
                padding: 0.75rem !important;
                font-size: 0.875rem !important;
            }
            
            /* File Upload - Clean */
            .stFileUploader {
                border: 1px dashed var(--border-light) !important;
                border-radius: 8px !important;
                padding: 1.5rem !important;
                text-align: center !important;
                background: var(--bg-secondary) !important;
                transition: all 0.15s ease !important;
            }
            
            .stFileUploader:hover {
                border-color: var(--accent-primary) !important;
                background: var(--info-bg) !important;
            }
            
            /* Headers - Clean */
            h1, h2, h3, h4, h5, h6 {
                color: var(--text-primary) !important;
                font-weight: 600 !important;
                transition: color 0.3s ease !important;
            }
            
            h1 { font-size: 1.5rem !important; }
            h2 { font-size: 1.25rem !important; }
            h3 { font-size: 1.125rem !important; }
            h4 { font-size: 1rem !important; }
            
            /* Subheaders */
            .stSubheader {
                font-size: 1rem !important;
                font-weight: 600 !important;
                color: var(--text-secondary) !important;
                margin-bottom: 0.75rem !important;
                transition: color 0.3s ease !important;
            }
            
            /* Welcome Message - ChatGPT Style */
            .welcome-message {
                text-align: center !important;
                padding: 3rem 2rem !important;
                color: var(--text-muted) !important;
                transition: color 0.3s ease !important;
            }
            
            .welcome-message h2 {
                font-size: 1.5rem !important;
                font-weight: 600 !important;
                color: var(--text-primary) !important;
                margin-bottom: 0.5rem !important;
                transition: color 0.3s ease !important;
            }
            
            .welcome-message p {
                font-size: 0.875rem !important;
                line-height: 1.6 !important;
                max-width: 500px !important;
                margin: 0 auto !important;
                transition: color 0.3s ease !important;
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
                .css-1d391kg {
                    position: relative !important;
                    width: 100% !important;
                    min-width: 100% !important;
                    max-width: 100% !important;
                    height: auto !important;
                    top: auto !important;
                    left: auto !important;
                }
                
                .block-container {
                    margin-left: 0 !important;
                    max-width: 100% !important;
                }
                
                .main-header h1 {
                    font-size: 1.25rem !important;
                }
                
                .chat-message {
                    padding: 1rem !important;
                }
                
                .message-content {
                    margin-left: 2rem !important;
                }
                
                .message-timestamp {
                    margin-left: 2rem !important;
                }
                
                .theme-toggle {
                    top: 0.5rem !important;
                    right: 0.5rem !important;
                    width: 35px !important;
                    height: 35px !important;
                    font-size: 1rem !important;
                }
            }
            
            /* Custom Scrollbar */
            ::-webkit-scrollbar {
                width: 6px !important;
            }
            
            ::-webkit-scrollbar-track {
                background: var(--bg-secondary) !important;
            }
            
            ::-webkit-scrollbar-thumb {
                background: var(--border-light) !important;
                border-radius: 3px !important;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: var(--text-muted) !important;
            }
            
            /* Remove any purple elements */
            [style*="purple"], [style*="#7f5af0"], [style*="#a4508b"], [style*="#5f0a87"] {
                color: var(--accent-primary) !important;
                background: var(--accent-primary) !important;
                border-color: var(--accent-primary) !important;
            }
            
            /* Override any existing purple gradients */
            [style*="linear-gradient"][style*="purple"], 
            [style*="linear-gradient"][style*="#7f5af0"], 
            [style*="linear-gradient"][style*="#a4508b"] {
                background: var(--accent-primary) !important;
            }
            
            /* Loading Animation */
            .loading-spinner {
                display: inline-block !important;
                width: 20px !important;
                height: 20px !important;
                border: 3px solid var(--border-light) !important;
                border-radius: 50% !important;
                border-top-color: var(--accent-primary) !important;
                animation: spin 1s ease-in-out infinite !important;
            }
            
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
            
            /* Progress Bar */
            .progress-bar {
                width: 100% !important;
                height: 4px !important;
                background: var(--bg-secondary) !important;
                border-radius: 2px !important;
                overflow: hidden !important;
                margin: 1rem 0 !important;
            }
            
            .progress-fill {
                height: 100% !important;
                background: var(--accent-primary) !important;
                border-radius: 2px !important;
                transition: width 0.3s ease !important;
            }
        </style>
    """
    return css