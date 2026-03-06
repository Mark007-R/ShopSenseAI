import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_recommendation_system import ProductRecommendationSystem
from config import DATA_PATH, DEFAULT_METHOD, DEFAULT_N_RECOMMENDATIONS, HYBRID_WEIGHTS, MIN_PURCHASE_THRESHOLD
import pandas as pd
import threading


class ModernRecommendationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Recommendation System - AI Powered")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        self.rec_system = None
        self.result = None
        self.current_user_data = None
        
        self.setup_styles()
        self.setup_ui()
        threading.Thread(target=self.load_system, daemon=True).start()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Segoe UI', 11), foreground='#34495e')
        style.configure('Header.TLabel', font=('Segoe UI', 10, 'bold'), foreground='#2c3e50')
        style.configure('Status.TLabel', font=('Segoe UI', 9), padding=5)
        
        style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'), padding=10)
        style.configure('Secondary.TButton', font=('Segoe UI', 9), padding=8)
        
        style.configure('Card.TFrame', background='#ffffff', relief='solid', borderwidth=1)
        style.configure('TNotebook', background='#ecf0f1')
        style.configure('TNotebook.Tab', font=('Segoe UI', 10), padding=[20, 10])
        
        self.root.configure(bg='#ecf0f1')
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg='#ecf0f1')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        header_frame = tk.Frame(main_container, bg='#2c3e50', height=100)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        header_frame.pack_propagate(False)
        
        title_label = ttk.Label(header_frame, text="🎯 Product Recommendation System", 
                               style='Title.TLabel', background='#2c3e50', foreground='white')
        title_label.pack(pady=(15, 5))
        
        subtitle_label = ttk.Label(header_frame, 
                                   text="AI-Powered Personalized Product Recommendations",
                                   style='Subtitle.TLabel', background='#2c3e50', foreground='#ecf0f1')
        subtitle_label.pack()
        
        status_frame = tk.Frame(main_container, bg='#34495e', height=40)
        status_frame.pack(fill=tk.X, pady=(0, 15))
        status_frame.pack_propagate(False)
        
        self.status_label = ttk.Label(status_frame, text="⏳ Loading system...", 
                                     style='Status.TLabel', background='#34495e', foreground='#f39c12')
        self.status_label.pack(side=tk.LEFT, padx=20, pady=5)
        
        self.stats_label = ttk.Label(status_frame, text="", 
                                     style='Status.TLabel', background='#34495e', foreground='#ecf0f1')
        self.stats_label.pack(side=tk.RIGHT, padx=20, pady=5)
        
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        self.setup_recommendation_tab(notebook)
        self.setup_batch_tab(notebook)
        self.setup_analytics_tab(notebook)
    
    def setup_recommendation_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(tab, text='🎁 Get Recommendations')
        
        content_frame = tk.Frame(tab, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        left_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=0, ipadx=15, ipady=15)
        
        ttk.Label(left_panel, text="Configuration", style='Header.TLabel', background='#ffffff').pack(anchor=tk.W, pady=(0, 15))
        
        user_type_frame = tk.LabelFrame(left_panel, text="User Type", bg='#ffffff', font=('Segoe UI', 9, 'bold'), 
                                       fg='#2c3e50', padx=10, pady=10)
        user_type_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.user_type_var = tk.StringVar(value="existing")
        tk.Radiobutton(user_type_frame, text="👤 Existing User", variable=self.user_type_var, 
                      value="existing", command=self.toggle_input_mode, bg='#ffffff',
                      font=('Segoe UI', 9), selectcolor='#3498db').pack(anchor=tk.W, pady=2)
        tk.Radiobutton(user_type_frame, text="🆕 New User", variable=self.user_type_var, 
                      value="new", command=self.toggle_input_mode, bg='#ffffff',
                      font=('Segoe UI', 9), selectcolor='#3498db').pack(anchor=tk.W, pady=2)
        
        self.user_input_frame = tk.LabelFrame(left_panel, text="User Input", bg='#ffffff', 
                                             font=('Segoe UI', 9, 'bold'), fg='#2c3e50', padx=10, pady=10)
        self.user_input_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.user_id_label = ttk.Label(self.user_input_frame, text="User ID:", background='#ffffff')
        self.user_id_label.pack(anchor=tk.W, pady=(0, 5))
        self.user_id_entry = ttk.Entry(self.user_input_frame, width=35, font=('Segoe UI', 9))
        self.user_id_entry.pack(fill=tk.X, pady=(0, 10))
        
        self.items_label = ttk.Label(self.user_input_frame, text="Items (comma-separated):", background='#ffffff')
        self.items_entry = ttk.Entry(self.user_input_frame, width=35, font=('Segoe UI', 9))
        
        method_frame = tk.LabelFrame(left_panel, text="Algorithm", bg='#ffffff', 
                                    font=('Segoe UI', 9, 'bold'), fg='#2c3e50', padx=10, pady=10)
        method_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.method_var = tk.StringVar(value=DEFAULT_METHOD)
        methods = [
            ("🤝 User-Based CF", "user_cf"),
            ("📦 Item-Based CF", "item_cf"),
            ("🎨 Content-Based", "content"),
            ("⚡ Hybrid (Best)", "hybrid")
        ]
        for text, value in methods:
            tk.Radiobutton(method_frame, text=text, variable=self.method_var, value=value,
                          bg='#ffffff', font=('Segoe UI', 9), selectcolor='#3498db').pack(anchor=tk.W, pady=2)
        
        params_frame = tk.LabelFrame(left_panel, text="Parameters", bg='#ffffff', 
                                    font=('Segoe UI', 9, 'bold'), fg='#2c3e50', padx=10, pady=10)
        params_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(params_frame, text="Recommendations:", background='#ffffff').pack(anchor=tk.W)
        self.n_recs_var = tk.StringVar(value=str(DEFAULT_N_RECOMMENDATIONS))
        ttk.Spinbox(params_frame, from_=1, to=50, textvariable=self.n_recs_var, 
                   width=33, font=('Segoe UI', 9)).pack(fill=tk.X, pady=(5, 0))
        
        button_frame = tk.Frame(left_panel, bg='#ffffff')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.generate_btn = tk.Button(button_frame, text="🚀 Generate", command=self.generate_recommendations,
                                     bg='#3498db', fg='white', font=('Segoe UI', 10, 'bold'),
                                     relief='flat', padx=20, pady=10, cursor='hand2', state='disabled')
        self.generate_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Button(button_frame, text="🗑️ Clear", command=self.clear_results,
                 bg='#95a5a6', fg='white', font=('Segoe UI', 10, 'bold'),
                 relief='flat', padx=20, pady=10, cursor='hand2').pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        right_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0, ipadx=15, ipady=15)
        
        results_header = tk.Frame(right_panel, bg='#ffffff')
        results_header.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(results_header, text="Recommendations", style='Header.TLabel', background='#ffffff').pack(side=tk.LEFT)
        
        self.save_btn = tk.Button(results_header, text="💾 Save CSV", command=self.save_to_csv,
                                 bg='#27ae60', fg='white', font=('Segoe UI', 9, 'bold'),
                                 relief='flat', padx=15, pady=5, cursor='hand2', state='disabled')
        self.save_btn.pack(side=tk.RIGHT)
        
        tree_frame = tk.Frame(right_panel, bg='#ffffff')
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar_y = tk.Scrollbar(tree_frame)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        scrollbar_x = tk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.results_tree = ttk.Treeview(tree_frame, 
                                        columns=('Rank', 'Product ID', 'Name', 'Category', 'Brand', 'Price', 'Rating', 'Score'),
                                        show='headings',
                                        yscrollcommand=scrollbar_y.set,
                                        xscrollcommand=scrollbar_x.set,
                                        height=20)
        
        scrollbar_y.config(command=self.results_tree.yview)
        scrollbar_x.config(command=self.results_tree.xview)
        
        columns_config = [
            ('Rank', 50),
            ('Product ID', 100),
            ('Name', 250),
            ('Category', 120),
            ('Brand', 120),
            ('Price', 80),
            ('Rating', 60),
            ('Score', 80)
        ]
        
        for col, width in columns_config:
            self.results_tree.heading(col, text=col, anchor=tk.W)
            self.results_tree.column(col, width=width, anchor=tk.W if col == 'Name' else tk.CENTER)
        
        self.results_tree.pack(fill=tk.BOTH, expand=True)
        
        self.results_tree.tag_configure('evenrow', background='#f8f9fa')
        self.results_tree.tag_configure('oddrow', background='#ffffff')
    
    def setup_batch_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(tab, text='📊 Batch Processing')
        
        content_frame = tk.Frame(tab, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        card = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        card.pack(fill=tk.BOTH, expand=True, padx=0, pady=0, ipadx=20, ipady=20)
        
        ttk.Label(card, text="Batch Recommendation Generator", style='Header.TLabel', background='#ffffff').pack(pady=(0, 20))
        
        input_frame = tk.Frame(card, bg='#ffffff')
        input_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(input_frame, text="Number of Users:", background='#ffffff', font=('Segoe UI', 9)).grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.batch_users_var = tk.StringVar(value="10")
        ttk.Spinbox(input_frame, from_=1, to=1000, textvariable=self.batch_users_var, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(input_frame, text="Recommendations per User:", background='#ffffff', font=('Segoe UI', 9)).grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.batch_recs_var = tk.StringVar(value="5")
        ttk.Spinbox(input_frame, from_=1, to=50, textvariable=self.batch_recs_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(input_frame, text="Method:", background='#ffffff', font=('Segoe UI', 9)).grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=5)
        self.batch_method_var = tk.StringVar(value="hybrid")
        ttk.Combobox(input_frame, textvariable=self.batch_method_var, 
                    values=["user_cf", "item_cf", "content", "hybrid"],
                    state="readonly", width=18).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        self.batch_progress = ttk.Progressbar(card, mode='indeterminate', length=400)
        self.batch_progress.pack(pady=(0, 20))
        
        self.batch_generate_btn = tk.Button(card, text="🚀 Generate Batch Recommendations",
                                           command=self.generate_batch,
                                           bg='#3498db', fg='white', font=('Segoe UI', 11, 'bold'),
                                           relief='flat', padx=30, pady=12, cursor='hand2', state='disabled')
        self.batch_generate_btn.pack(pady=(0, 20))
        
        self.batch_status = ttk.Label(card, text="", background='#ffffff', font=('Segoe UI', 10))
        self.batch_status.pack()
    
    def setup_analytics_tab(self, notebook):
        tab = tk.Frame(notebook, bg='#ecf0f1')
        notebook.add(tab, text='📈 Analytics')
        
        content_frame = tk.Frame(tab, bg='#ecf0f1')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        card = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        card.pack(fill=tk.BOTH, expand=True, ipadx=20, ipady=20)
        
        ttk.Label(card, text="System Statistics", style='Header.TLabel', background='#ffffff').pack(pady=(0, 20))
        
        self.analytics_text = tk.Text(card, wrap=tk.WORD, font=('Courier New', 10), 
                                     height=25, bg='#f8f9fa', relief='flat', padx=20, pady=20)
        self.analytics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    
    def toggle_input_mode(self):
        if self.user_type_var.get() == "new":
            self.user_id_label.pack_forget()
            self.user_id_entry.pack_forget()
            self.items_label.pack(anchor=tk.W, pady=(0, 5))
            self.items_entry.pack(fill=tk.X, pady=(0, 10))
        else:
            self.items_label.pack_forget()
            self.items_entry.pack_forget()
            self.user_id_label.pack(anchor=tk.W, pady=(0, 5))
            self.user_id_entry.pack(fill=tk.X, pady=(0, 10))
    
    def load_system(self):
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    DATA_PATH.replace("/", os.sep))
            self.rec_system = ProductRecommendationSystem(
                data_path=data_path,
                min_purchase_threshold=MIN_PURCHASE_THRESHOLD
            )
            
            self.root.after(0, lambda: self.status_label.config(
                text="✅ System Ready", foreground='#27ae60'))
            
            users = self.rec_system.data['user_id'].nunique()
            products = self.rec_system.data['product_id'].nunique()
            interactions = len(self.rec_system.data)
            
            self.root.after(0, lambda: self.stats_label.config(
                text=f"👥 {users} Users | 📦 {products} Products | 🔗 {interactions:,} Interactions"))
            
            self.root.after(0, lambda: self.generate_btn.config(state='normal'))
            self.root.after(0, lambda: self.batch_generate_btn.config(state='normal'))
            
            sample_user = self.rec_system.data['user_id'].iloc[0]
            self.root.after(0, lambda: self.user_id_entry.insert(0, sample_user))
            
            self.update_analytics()
            
        except Exception as e:
            self.root.after(0, lambda: self.status_label.config(
                text="❌ Error Loading System", foreground='#e74c3c'))
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load: {str(e)}"))
    
    def generate_recommendations(self):
        try:
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            n_recs = int(self.n_recs_var.get())
            
            if self.user_type_var.get() == "new":
                items_input = self.items_entry.get().strip()
                if not items_input:
                    messagebox.showwarning("Input Required", "Please enter item preferences")
                    return
                seed_items = [item.strip() for item in items_input.split(",")]
                self.result = self.rec_system.recommend_for_new_user(seed_items=seed_items, n_recommendations=n_recs)
            else:
                user_id = self.user_id_entry.get().strip()
                if not user_id:
                    messagebox.showwarning("Input Required", "Please enter a user ID")
                    return
                
                method = self.method_var.get()
                kwargs = {"weights": HYBRID_WEIGHTS} if method == "hybrid" else {}
                self.result = self.rec_system.get_recommendations(
                    user_id=user_id, method=method, n_recommendations=n_recs, **kwargs
                )
            
            if not self.result['recommendations']:
                messagebox.showinfo("No Results", "No recommendations found")
                return
            
            for idx, rec in enumerate(self.result['recommendations'], 1):
                tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
                values = (
                    idx,
                    rec.get('product_id', ''),
                    rec.get('product_name', '')[:50],
                    rec.get('category', ''),
                    rec.get('brand', ''),
                    rec.get('price', ''),
                    f"{rec.get('rating', 0):.1f}⭐",
                    f"{rec.get('score', 0):.4f}"
                )
                self.results_tree.insert('', tk.END, values=values, tags=(tag,))
            
            self.save_btn.config(state='normal')
            messagebox.showinfo("Success", f"Generated {len(self.result['recommendations'])} recommendations!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate: {str(e)}")
    
    def clear_results(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.result = None
        self.save_btn.config(state='disabled')
    
    def save_to_csv(self):
        if not self.result:
            messagebox.showwarning("No Data", "No recommendations to save")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile="recommendations.csv"
            )
            
            if filename:
                df = pd.DataFrame(self.result.get('recommendations', []))
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def generate_batch(self):
        def run():
            try:
                self.batch_progress.start()
                self.batch_generate_btn.config(state='disabled')
                self.batch_status.config(text="Processing...", foreground='#f39c12')
                
                n_users = int(self.batch_users_var.get())
                n_recs = int(self.batch_recs_var.get())
                method = self.batch_method_var.get()
                
                user_ids = self.rec_system.data['user_id'].unique()[:n_users]
                
                output_df = self.rec_system.generate_batch_recommendations(
                    user_ids=user_ids.tolist(),
                    method=method,
                    n_recommendations=n_recs
                )
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")],
                    initialfile=f"batch_recommendations_{n_users}users.csv"
                )
                
                if filename:
                    output_df.to_csv(filename, index=False)
                    self.root.after(0, lambda: self.batch_status.config(
                        text=f"✅ Saved {len(output_df)} recommendations to file!", foreground='#27ae60'))
                    self.root.after(0, lambda: messagebox.showinfo("Success", 
                        f"Generated {len(output_df)} recommendations for {n_users} users!"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Batch processing failed: {str(e)}"))
                self.root.after(0, lambda: self.batch_status.config(text="❌ Error", foreground='#e74c3c'))
            finally:
                self.root.after(0, lambda: self.batch_progress.stop())
                self.root.after(0, lambda: self.batch_generate_btn.config(state='normal'))
        
        threading.Thread(target=run, daemon=True).start()
    
    def update_analytics(self):
        if not self.rec_system:
            return
        
        try:
            data = self.rec_system.data
            
            analytics_text = "=" * 80 + "\n"
            analytics_text += "                    SYSTEM ANALYTICS DASHBOARD\n"
            analytics_text += "=" * 80 + "\n\n"
            
            analytics_text += "📊 DATASET OVERVIEW\n"
            analytics_text += "-" * 80 + "\n"
            analytics_text += f"  Total Records:          {len(data):,}\n"
            analytics_text += f"  Unique Users:           {data['user_id'].nunique():,}\n"
            analytics_text += f"  Unique Products:        {data['product_id'].nunique():,}\n"
            analytics_text += f"  Unique Categories:      {data['category'].nunique():,}\n"
            analytics_text += f"  Unique Brands:          {data['brand'].nunique():,}\n\n"
            
            analytics_text += "🛒 INTERACTION TYPES\n"
            analytics_text += "-" * 80 + "\n"
            interaction_counts = data['interaction_type'].value_counts()
            for interaction, count in interaction_counts.items():
                percentage = (count / len(data)) * 100
                analytics_text += f"  {interaction.ljust(20)}: {count:>8,} ({percentage:>5.1f}%)\n"
            analytics_text += "\n"
            
            analytics_text += "📁 TOP 10 CATEGORIES\n"
            analytics_text += "-" * 80 + "\n"
            top_categories = data['category'].value_counts().head(10)
            for idx, (category, count) in enumerate(top_categories.items(), 1):
                analytics_text += f"  {idx:>2}. {category.ljust(30)}: {count:>6,} interactions\n"
            analytics_text += "\n"
            
            analytics_text += "🏢 TOP 10 BRANDS\n"
            analytics_text += "-" * 80 + "\n"
            top_brands = data['brand'].value_counts().head(10)
            for idx, (brand, count) in enumerate(top_brands.items(), 1):
                analytics_text += f"  {idx:>2}. {brand.ljust(30)}: {count:>6,} interactions\n"
            analytics_text += "\n"
            
            analytics_text += "⭐ RATING STATISTICS\n"
            analytics_text += "-" * 80 + "\n"
            analytics_text += f"  Average Rating:         {data['rating'].mean():.2f} / 5.0\n"
            analytics_text += f"  Median Rating:          {data['rating'].median():.2f}\n"
            analytics_text += f"  Min Rating:             {data['rating'].min():.2f}\n"
            analytics_text += f"  Max Rating:             {data['rating'].max():.2f}\n\n"
            
            analytics_text += "💰 PRICE STATISTICS (INR)\n"
            analytics_text += "-" * 80 + "\n"
            analytics_text += f"  Average Price:          ₹{data['listed_price_inr'].mean():,.2f}\n"
            analytics_text += f"  Median Price:           ₹{data['listed_price_inr'].median():,.2f}\n"
            analytics_text += f"  Min Price:              ₹{data['listed_price_inr'].min():,.2f}\n"
            analytics_text += f"  Max Price:              ₹{data['listed_price_inr'].max():,.2f}\n\n"
            
            analytics_text += "👥 USER SEGMENTS\n"
            analytics_text += "-" * 80 + "\n"
            if 'user_segment' in data.columns:
                segments = data.groupby('user_segment')['user_id'].nunique()
                for segment, count in segments.items():
                    analytics_text += f"  {segment.ljust(30)}: {count:>6,} users\n"
            analytics_text += "\n"
            
            analytics_text += "=" * 80 + "\n"
            
            self.analytics_text.delete(1.0, tk.END)
            self.analytics_text.insert(1.0, analytics_text)
            
        except Exception as e:
            self.analytics_text.insert(1.0, f"Error generating analytics: {str(e)}")


def main():
    root = tk.Tk()
    app = ModernRecommendationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
