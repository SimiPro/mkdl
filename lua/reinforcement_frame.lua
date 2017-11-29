--[[
Play.lua

This script is used to actually play the Convolutional AI on a track. It asynchronously communicates
with predict-server.py over a TCP socket. The message protocol is a simple, line-oriented feed.
This module can also be called as a function, in which case the first argument is the number of
frames to play for.
]]--

--[[ BEGIN CONFIGURATION ]]--
USE_CLIPBOARD = true -- Use the clipboard to send screenshots to the predict server.

--[[ How many frames to wait before sending a new prediction request. If you're using a file, you
may want to consider adding some frames here. ]]--
WAIT_FRAMES = 5

USE_MAPPING = true -- Whether or not to use input remapping.
CHECK_PROGRESS_EVERY = 300 -- Check progress after this many frames to detect if we get stuck.
--[[ END CONFIGURATION ]]--

savestate.loadslot(2)
savestate.saveslot(2) -- save current slot for reset purposes

local chunk_args = {...}
local PLAY_FOR_FRAMES = chunk_args[1]
if PLAY_FOR_FRAMES ~= nil then print("Playing for " .. PLAY_FOR_FRAMES .. " frames.") end

local util = require("util")

local SCREENSHOT_FILE = util.getTMPDir() .. '\\predict-screenshot.png'

local tcp = require("lualibs.socket").tcp()
local success, error = tcp:connect('localhost', 36296)
if not success then
  print("Failed to connect to server:", error)
  return
end

client.setscreenshotosd(false)

local course = util.readCourse()
local reward = util.readProgress()

--- inform mario environment that we're ready
-- tcp:send("STARTED\n")
tcp:settimeout(3)

outgoing_message, outgoing_message_index = nil, nil
function request_prediction()
  reward = util.readProgress()
  if USE_CLIPBOARD then
    client.screenshottoclipboard()
    outgoing_message = "MESSAGE screenshot:clip:reward:" .. reward .. ":done:False \n"
  else
    client.screenshot(SCREENSHOT_FILE)
    outgoing_message = "MESSAGE screenshot:" .. SCREENSHOT_FILE .. ":reward:" .. reward .. ":done:False \n"
    --outgoing_message = "PREDICT:" .. SCREENSHOT_FILE .. "\n"
  end
  outgoing_message_index = 1
end
request_prediction()

local receive_buffer = ""

function onexit()
  client.pause()
  tcp:close()
end
local exit_guid = event.onexit(onexit)

local current_action = 0
local frame = 1
local max_progress = util.readProgress()
local esc_prev = input.get()['Escape']

BOX_CENTER_X, BOX_CENTER_Y = 160, 215
BOX_WIDTH, BOX_HEIGHT = 100, 4
SLIDER_WIDTH, SLIDER_HIEGHT = 4, 16
function draw_info()
  gui.drawBox(BOX_CENTER_X - BOX_WIDTH / 2, BOX_CENTER_Y - BOX_HEIGHT / 2,
              BOX_CENTER_X + BOX_WIDTH / 2, BOX_CENTER_Y + BOX_HEIGHT / 2,
              none, 0x60FFFFFF)
  gui.drawBox(BOX_CENTER_X + current_action*(BOX_WIDTH / 2) - SLIDER_WIDTH / 2, BOX_CENTER_Y - SLIDER_HIEGHT / 2,
              BOX_CENTER_X + current_action*(BOX_WIDTH / 2) + SLIDER_WIDTH / 2, BOX_CENTER_Y + SLIDER_HIEGHT / 2,
              none, 0xFFFF0000)
end

while util.readProgress() < 3 do -- 3 means 3 laps

  -- Process the outgoing message.
  if outgoing_message ~= nil then
    local sent, error, last_byte = tcp:send(outgoing_message, outgoing_message_index)
    if sent ~= nil then
      outgoing_message = nil
      outgoing_message_index = nil
    else
      if error == "timeout" then
        outgoing_message_index = last_byte + 1
      else
        print("Send failed: ", error); break
      end
    end
  end

 -- process incoming message
  local message, error, waited
  message = nil
  waited = 0
  while message == nil and waited < 1000  do
    waited = waited + 1
    message, error, receive_buffer = tcp:receive("*l", receive_buffer)
  end

  if message == nil then
    if error ~= "timeout" then
      print("Receive failed: ", error); break
    end
  end
  if waited == 1000 then
    break
  end


  local m_worked = 0
  if message == "RESET" then
    console.log('Reset game - LOADING SLOT 2 Which we saved at the beginning')
    savestate.loadslot(2)
    client.unpause()
  elseif message ~= "PREDICTIONERROR" then
    current_action = tonumber(message)
    for i=1, WAIT_FRAMES do
      joypad.set({["P1 A"] = true})
      joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(current_action) })
      draw_info()
      emu.frameadvance()
    end
  else
    print("Prediction error...")
  end

  -- inform serer about current state and request action
  request_prediction()
  
  joypad.set({["P1 A"] = true})
  joypad.setanalog({["P1 X Axis"] = util.convertSteerToJoystick(current_action) })
  draw_info()
  emu.frameadvance()

  -- If we haven't made any progress since the last check, just break.
  if frame % CHECK_PROGRESS_EVERY == 0 then
    if util.readProgress() <= max_progress then
      break
    else max_progress = util.readProgress() end
  end

  if not esc_prev and input.get()['Escape'] then break end
  esc_prev = input.get()['Escape']
end

onexit()
event.unregisterbyid(exit_guid)
