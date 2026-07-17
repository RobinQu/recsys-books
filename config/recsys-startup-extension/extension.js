const vscode = require('vscode');

async function activate() {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) return;
  const train = vscode.Uri.joinPath(folder.uri, 'train.py');
  try {
    await vscode.workspace.fs.stat(train);
    const document = await vscode.workspace.openTextDocument(train);
    await vscode.window.showTextDocument(document, { preview: false });
  } catch {
    // Foundation workspaces without train.py remain empty by design.
  }
}

module.exports = { activate };
