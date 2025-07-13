#!/usr/bin/env python3
"""
Create sample SWTBot training data if none exists.
This allows you to test training without collecting data first.
"""

import json
import os
from pathlib import Path

def create_sample_swtbot_data():
    """Create sample SWTBot Java code for training."""
    
    sample_files = [
        {
            "file_path": "sample1.java",
            "content": """import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotButton;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotText;
import org.junit.Test;
import static org.junit.Assert.*;

public class ButtonClickTest {
    
    @Test
    public void testButtonClick() {
        SWTBot bot = new SWTBot();
        
        // Find and click the Submit button
        SWTBotButton submitButton = bot.button("Submit");
        assertTrue("Submit button should be enabled", submitButton.isEnabled());
        submitButton.click();
        
        // Verify button was clicked
        assertFalse("Submit button should be disabled after click", submitButton.isEnabled());
    }
}""",
            "metadata": {
                "file_name": "ButtonClickTest.java",
                "line_count": 18,
                "char_count": 567,
                "swtbot_keywords": ["SWTBot", "SWTBotButton", "button", "click"],
                "imports": ["org.eclipse.swtbot.swt.finder.SWTBot", "org.junit.Test"],
                "package": None,
                "classes": ["ButtonClickTest"],
                "methods": 1
            },
            "hash": "abc123"
        },
        {
            "file_path": "sample2.java", 
            "content": """import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotText;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotButton;
import org.junit.Test;
import static org.junit.Assert.*;

public class TextInputTest {
    
    @Test
    public void testTextInput() {
        SWTBot bot = new SWTBot();
        
        // Find text field and enter text
        SWTBotText textField = bot.textWithLabel("Name:");
        textField.setText("John Doe");
        assertEquals("John Doe", textField.getText());
        
        // Click OK button
        SWTBotButton okButton = bot.button("OK");
        okButton.click();
    }
}""",
            "metadata": {
                "file_name": "TextInputTest.java",
                "line_count": 19,
                "char_count": 612,
                "swtbot_keywords": ["SWTBot", "SWTBotText", "SWTBotButton", "textWithLabel", "setText"],
                "imports": ["org.eclipse.swtbot.swt.finder.SWTBot", "org.junit.Test"],
                "package": None,
                "classes": ["TextInputTest"],
                "methods": 1
            },
            "hash": "def456"
        },
        {
            "file_path": "sample3.java",
            "content": """import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotTree;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotTreeItem;
import org.junit.Test;
import static org.junit.Assert.*;

public class TreeNavigationTest {
    
    @Test
    public void testTreeNavigation() {
        SWTBot bot = new SWTBot();
        
        // Find tree widget
        SWTBotTree tree = bot.tree();
        
        // Expand root node
        SWTBotTreeItem rootItem = tree.getTreeItem("Root");
        rootItem.expand();
        
        // Select child item
        SWTBotTreeItem childItem = rootItem.getNode("Child 1");
        childItem.select();
        
        assertTrue("Child item should be selected", childItem.isSelected());
    }
}""",
            "metadata": {
                "file_name": "TreeNavigationTest.java",
                "line_count": 22,
                "char_count": 743,
                "swtbot_keywords": ["SWTBot", "SWTBotTree", "SWTBotTreeItem", "tree", "expand", "select"],
                "imports": ["org.eclipse.swtbot.swt.finder.SWTBot", "org.junit.Test"],
                "package": None,
                "classes": ["TreeNavigationTest"],
                "methods": 1
            },
            "hash": "ghi789"
        },
        {
            "file_path": "sample4.java",
            "content": """import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotTable;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotTableItem;
import org.junit.Test;
import static org.junit.Assert.*;

public class TableInteractionTest {
    
    @Test
    public void testTableInteraction() {
        SWTBot bot = new SWTBot();
        
        // Find table widget
        SWTBotTable table = bot.table();
        
        // Select first row
        SWTBotTableItem firstRow = table.getTableItem(0);
        firstRow.select();
        
        // Verify selection
        assertTrue("First row should be selected", firstRow.isSelected());
        
        // Double-click to open
        firstRow.doubleClick();
    }
}""",
            "metadata": {
                "file_name": "TableInteractionTest.java",
                "line_count": 23,
                "char_count": 756,
                "swtbot_keywords": ["SWTBot", "SWTBotTable", "SWTBotTableItem", "table", "select", "doubleClick"],
                "imports": ["org.eclipse.swtbot.swt.finder.SWTBot", "org.junit.Test"],
                "package": None,
                "classes": ["TableInteractionTest"],
                "methods": 1
            },
            "hash": "jkl012"
        },
        {
            "file_path": "sample5.java",
            "content": """import org.eclipse.swtbot.swt.finder.SWTBot;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotMenu;
import org.eclipse.swtbot.swt.finder.widgets.SWTBotShell;
import org.junit.Test;
import static org.junit.Assert.*;

public class MenuOperationTest {
    
    @Test
    public void testMenuOperation() {
        SWTBot bot = new SWTBot();
        
        // Access File menu
        SWTBotMenu fileMenu = bot.menu("File");
        fileMenu.click();
        
        // Click New submenu
        SWTBotMenu newMenu = bot.menu("New");
        newMenu.click();
        
        // Verify dialog opened
        SWTBotShell dialog = bot.shell("New File");
        assertTrue("New File dialog should be active", dialog.isActive());
        
        // Close dialog
        dialog.close();
    }
}""",
            "metadata": {
                "file_name": "MenuOperationTest.java",
                "line_count": 25,
                "char_count": 812,
                "swtbot_keywords": ["SWTBot", "SWTBotMenu", "SWTBotShell", "menu", "shell", "click"],
                "imports": ["org.eclipse.swtbot.swt.finder.SWTBot", "org.junit.Test"],
                "package": None,
                "classes": ["MenuOperationTest"],
                "methods": 1
            },
            "hash": "mno345"
        }
    ]
    
    # Duplicate the samples to have more training data
    extended_samples = []
    for i in range(20):  # Create 100 samples total (5 * 20)
        for sample in sample_files:
            new_sample = sample.copy()
            new_sample["file_path"] = f"sample_{i}_{sample['file_path']}"
            new_sample["hash"] = f"{sample['hash']}_{i}"
            extended_samples.append(new_sample)
    
    return extended_samples

def save_sample_data():
    """Save sample data to the expected location."""
    
    # Create directories
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample data
    sample_data = create_sample_swtbot_data()
    
    # Save as JSONL file
    output_file = processed_dir / "processed_java_files.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in sample_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # Create processing stats
    stats = {
        "files_processed": len(sample_data),
        "files_accepted": len(sample_data),
        "files_rejected": 0,
        "rejection_reasons": {},
        "total_lines": sum(sample["metadata"]["line_count"] for sample in sample_data),
        "total_characters": sum(sample["metadata"]["char_count"] for sample in sample_data)
    }
    
    stats_file = processed_dir / "processing_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Created {len(sample_data)} sample training files")
    print(f"✅ Saved to {output_file}")
    print(f"✅ Total lines: {stats['total_lines']:,}")
    print(f"✅ Total characters: {stats['total_characters']:,}")
    
    return output_file

def main():
    """Main function."""
    print("Creating sample SWTBot training data...")
    
    # Check if data already exists
    data_file = Path("data/processed/processed_java_files.jsonl")
    if data_file.exists():
        print(f"✅ Training data already exists at {data_file}")
        
        # Count existing samples
        with open(data_file, 'r', encoding='utf-8') as f:
            count = sum(1 for line in f)
        print(f"✅ Found {count} existing training samples")
        
        response = input("Do you want to recreate the sample data? (y/N): ")
        if response.lower() != 'y':
            print("Using existing data.")
            return
    
    # Create sample data
    save_sample_data()
    print("\n🎉 Sample data created successfully!")
    print("\nYou can now run training with:")
    print("python scripts/train.py --batch-size 1 --max-steps 500")

if __name__ == "__main__":
    main()
