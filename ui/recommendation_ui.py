import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from product_recommendation_system import ProductRecommendationSystem
from config import DATA_PATH, DEFAULT_METHOD, DEFAULT_N_RECOMMENDATIONS, HYBRID_WEIGHTS, MIN_PURCHASE_THRESHOLD


class RecommendationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Product Recommendation System")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.rec_system = None
        self.result = None
        
        self.setup_ui()
        self.load_system()
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        title_label = ttk.Label(main_frame, text="Product Recommendation System", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        status_frame = ttk.LabelFrame(main_frame, text="System Status", padding="5")
        status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.status_label = ttk.Label(status_frame, text="Loading...", foreground="orange")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(input_frame, text="User Type:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.user_type_var = tk.StringVar(value="existing")
        ttk.Radiobutton(input_frame, text="Existing User", variable=self.user_type_var, 
                       value="existing", command=self.toggle_input_mode).grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(input_frame, text="New User", variable=self.user_type_var, 
                       value="new", command=self.toggle_input_mode).grid(row=0, column=2, sticky=tk.W)
        
        self.user_id_label = ttk.Label(input_frame, text="User ID:")
        self.user_id_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        self.user_id_entry = ttk.Entry(input_frame, width=30)
        self.user_id_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.items_label = ttk.Label(input_frame, text="Items (comma-separated):")
        self.items_entry = ttk.Entry(input_frame, width=30)
        
        ttk.Label(input_frame, text="Method:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.method_var = tk.StringVar(value=DEFAULT_METHOD)
        method_combo = ttk.Combobox(input_frame, textvariable=self.method_var, 
                                    values=["user_cf", "item_cf", "content", "hybrid"],
                                    state="readonly", width=28)
        method_combo.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(input_frame, text="Number of Recommendations:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.n_recs_var = tk.StringVar(value=str(DEFAULT_N_RECOMMENDATIONS))
        self.n_recs_spinbox = ttk.Spinbox(input_frame, from_=1, to=50, textvariable=self.n_recs_var, width=28)
        self.n_recs_spinbox.grid(row=3, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.generate_btn = ttk.Button(button_frame, text="Generate Recommendations", 
                                       command=self.generate_recommendations, state="disabled")
        self.generate_btn.grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, text="Clear", command=self.clear_results).grid(row=0, column=1, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save to CSV", 
                                   command=self.save_to_csv, state="disabled")
        self.save_btn.grid(row=0, column=2, padx=5)
        
        results_frame = ttk.LabelFrame(main_frame, text="Recommendations", padding="10")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        main_frame.rowconfigure(4, weight=1)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, width=100, height=25, 
                                                      wrap=tk.WORD, font=("Courier", 9))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
        for child in main_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                for subchild in child.winfo_children():
                    subchild.grid_configure(padx=5)
    
    def toggle_input_mode(self):
        if self.user_type_var.get() == "new":
            self.user_id_label.grid_remove()
            self.user_id_entry.grid_remove()
            self.items_label.grid(row=1, column=0, sticky=tk.W, pady=5)
            self.items_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        else:
            self.items_label.grid_remove()
            self.items_entry.grid_remove()
            self.user_id_label.grid(row=1, column=0, sticky=tk.W, pady=5)
            self.user_id_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
    
    def load_system(self):
        self.results_text.insert(tk.END, "Loading recommendation system...\n")
        self.root.update()
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                    DATA_PATH.replace("/", os.sep))
            self.rec_system = ProductRecommendationSystem(
                data_path=data_path,
                min_purchase_threshold=MIN_PURCHASE_THRESHOLD
            )
            self.status_label.config(text="System Ready", foreground="green")
            self.generate_btn.config(state="normal")
            
            sample_user = self.rec_system.data['user_id'].iloc[0]
            self.user_id_entry.insert(0, sample_user)
            
            self.results_text.insert(tk.END, f"System loaded successfully!\n")
            self.results_text.insert(tk.END, f"Total users: {self.rec_system.data['user_id'].nunique()}\n")
            self.results_text.insert(tk.END, f"Total products: {self.rec_system.data['product_id'].nunique()}\n")
            self.results_text.insert(tk.END, f"Total interactions: {len(self.rec_system.data)}\n\n")
        except Exception as e:
            self.status_label.config(text="Error Loading System", foreground="red")
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
            messagebox.showerror("Error", f"Failed to load system: {str(e)}")
    
    def generate_recommendations(self):
        try:
            self.results_text.delete(1.0, tk.END)
            n_recs = int(self.n_recs_var.get())
            
            if self.user_type_var.get() == "new":
                items_input = self.items_entry.get().strip()
                if not items_input:
                    messagebox.showwarning("Input Required", "Please enter item preferences")
                    return
                seed_items = [item.strip() for item in items_input.split(",")]
                self.result = self.rec_system.recommend_for_new_user(
                    seed_items=seed_items,
                    n_recommendations=n_recs
                )
                self.results_text.insert(tk.END, f"New User Recommendations\n")
                self.results_text.insert(tk.END, f"Input items: {seed_items}\n")
                self.results_text.insert(tk.END, f"Matched terms: {self.result.get('matched_items', [])}\n\n")
            else:
                user_id = self.user_id_entry.get().strip()
                if not user_id:
                    messagebox.showwarning("Input Required", "Please enter a user ID")
                    return
                
                method = self.method_var.get()
                kwargs = {"weights": HYBRID_WEIGHTS} if method == "hybrid" else {}
                
                self.result = self.rec_system.get_recommendations(
                    user_id=user_id,
                    method=method,
                    n_recommendations=n_recs,
                    **kwargs
                )
                self.results_text.insert(tk.END, f"Recommendations for User: {user_id}\n")
                self.results_text.insert(tk.END, f"Method: {method}\n\n")
            
            if not self.result['recommendations']:
                self.results_text.insert(tk.END, "No recommendations found.\n")
                return
            
            self.results_text.insert(tk.END, "=" * 100 + "\n")
            self.results_text.insert(tk.END, f"{'Rank':<6}{'Product ID':<15}{'Product Name':<40}{'Category':<20}{'Score':<10}\n")
            self.results_text.insert(tk.END, "-" * 100 + "\n")
            
            for idx, rec in enumerate(self.result['recommendations'], 1):
                product_name = str(rec.get('product_name', ''))[:38]
                category = str(rec.get('category', ''))[:18]
                score = rec.get('score', 0)
                product_id = str(rec.get('product_id', ''))[:13]
                
                line = f"{idx:<6}{product_id:<15}{product_name:<40}{category:<20}{score:<10.4f}\n"
                self.results_text.insert(tk.END, line)
            
            self.results_text.insert(tk.END, "=" * 100 + "\n")
            self.results_text.insert(tk.END, f"\nTotal recommendations: {len(self.result['recommendations'])}\n")
            
            self.save_btn.config(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate recommendations: {str(e)}")
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
    
    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.result = None
        self.save_btn.config(state="disabled")
    
    def save_to_csv(self):
        if not self.result:
            messagebox.showwarning("No Data", "No recommendations to save")
            return
        
        try:
            import pandas as pd
            from tkinter import filedialog
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile="recommendations.csv"
            )
            
            if filename:
                df = pd.DataFrame(self.result.get('recommendations', []))
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Recommendations saved to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")


def main():
    root = tk.Tk()
    app = RecommendationUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
