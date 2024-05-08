const express = require("express");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = 3000;
const IP = "localhost";

// Set the view engine to EJS
app.set("view engine", "ejs");

// Serve static files from the 'public' directory
app.use(express.static(path.join(__dirname, "public")));

// Define a route for the homepage
app.get("/critic", (req, res) => {
  fs.readFile("critic_data.json", "utf8", (err, data) => {
    if (err) {
      console.error("Error reading file:", err);
      res.status(500).send("Internal Server Error");
      return;
    }

    // Parse JSON data
    const jsonData = JSON.parse(data);

    // Render the EJS template with the JSON data
    res.render("index", { jsonData, type: "Critic Only" });
  });
});

app.get("/actor-critic", (req, res) => {
  fs.readFile("actorcritic_data.json", "utf8", (err, data) => {
    if (err) {
      console.error("Error reading file:", err);
      res.status(500).send("Internal Server Error");
      return;
    }

    // Parse JSON data
    const jsonData = JSON.parse(data);

    // Render the EJS template with the JSON data
    res.render("index", { jsonData, type: "Actor Critic" });
  });
});

// Start the server
app.listen(PORT, IP, () => {
  console.log(`Server running at http://${IP}:${PORT}/`);
});
